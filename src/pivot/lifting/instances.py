"""Recover stationary instances when a SAM track spans several pieces of furniture."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import footprint_overlap
from pivot.geometry.types import BoolArray, FloatArray
from pivot.lifting.backproject import masked_points
from pivot.lifting.objects import distinct_pairs, duplicate_groups, largest_cluster, split_overlaps
from pivot.lifting.objects import voxel_votes
from pivot.perception.tracks import Track
from pivot.reconstruction.types import Reconstruction


@dataclass(frozen=True)
class InstanceConfig:
    """Bounds for conservative same-category instance recovery."""

    voxel: float
    min_points: int
    extent: float
    overlap: float
    samples: int = 16


@dataclass
class _Candidate:
    cloud: FloatArray
    sources: set[int]
    observations: dict[int,FloatArray]


def _sample_views(recon: Reconstruction, track: Track, config: InstanceConfig
                  ) -> list[tuple[int,FloatArray]]:
    """Samples compact connected observations without fusing a drifting identity."""
    frames = sorted(track.masks)
    views = []
    for at in np.linspace(0,len(frames)-1,min(config.samples,len(frames)),dtype=int):
        cloud = masked_points(recon,frames[at],track.masks[frames[at]])
        if len(cloud)<config.min_points:
            continue
        cloud = largest_cluster(voxel_votes([cloud],config.voxel)[0],1.5*config.voxel)
        if len(cloud)>=config.min_points:
            views.append((frames[at],cloud))
    return views


def _wide(cloud: FloatArray, views: list[tuple[int,FloatArray]],
          fit: Callable[[FloatArray],OrientedBox], extent: float) -> bool:
    """Detects spatial drift even if largest-component fusion discarded one instance."""
    compact = [c for _,c in views if max(fit(c).size[:2])<=extent]
    return bool(max(fit(cloud).size[:2])>extent or len(compact)>=2
                and max(fit(np.vstack(compact)).size[:2])>extent)


def _inside(points: FloatArray, box: OrientedBox, margin: float) -> BoolArray:
    """Selects observed points near a footprint, without inventing its surface."""
    result: BoolArray = np.all(np.abs(box.to_local(points)[:,:2]) <= box.size[:2]/2+margin,axis=1)
    return result


def _seeds(tracks: list[Track], clouds: list[FloatArray], fit: Callable[[FloatArray],OrientedBox],
           config: InstanceConfig, views: list[list[tuple[int,FloatArray]]]
           ) -> tuple[list[list[int]], list[FloatArray], list[int]]:
    """Separates compact reliable tracks from multi-instance observations."""
    compact = [i for i,c in enumerate(clouds) if len(c)>=config.min_points
               and not _wide(c,views[i],fit,config.extent)]
    broad = [i for i,c in enumerate(clouds) if len(c)>=config.min_points
             and i not in compact]
    if not compact or not broad:
        return [],[],broad
    groups = duplicate_groups([clouds[i] for i in compact],1.5*config.voxel,config.overlap,
        [fit(clouds[i]) for i in compact],distinct=distinct_pairs([tracks[i] for i in compact]))
    groups = [[compact[i] for i in g] for g in groups]
    return groups,[np.unique(np.vstack([clouds[i] for i in g]),axis=0) for g in groups],broad


def _view_candidates(views: list[list[tuple[int,FloatArray]]], broad: list[int],
                     seeds: list[FloatArray], fit: Callable[[FloatArray],OrientedBox],
                     config: InstanceConfig) -> list[_Candidate]:
    """Uses compact single-view observations to recover missing spatial instances."""
    boxes = [fit(c) for c in seeds]
    minimum = .5*np.median([np.sort(b.size[:2]) for b in boxes],axis=0)
    candidates: list[_Candidate] = []
    for index in broad:
        for frame,cloud in views[index]:
            box = fit(cloud)
            if max(box.size[:2])>config.extent or (np.sort(box.size[:2])<minimum).any() \
                    or any(footprint_overlap(box,b)>.6 for b in boxes):
                continue
            _add_candidate(candidates,cloud,index,frame,fit,config.extent)
    # A single drifting track is insufficient evidence for a new instance.
    return _confirmed(candidates,config)


def _confirmed(candidates: list[_Candidate], config: InstanceConfig) -> list[_Candidate]:
    """Requires independent source identities and repeat observations of each retained voxel."""
    confirmed = []
    for candidate in candidates:
        if len(candidate.sources)<2 or len(candidate.observations)<2:
            continue
        centers,votes = voxel_votes(list(candidate.observations.values()),config.voxel)
        candidate.cloud = largest_cluster(centers[votes>=2],1.5*config.voxel)
        if len(candidate.cloud)>=config.min_points:
            confirmed.append(candidate)
    return confirmed


def _add_candidate(candidates: list[_Candidate], cloud: FloatArray, source: int, frame: int,
                   fit: Callable[[FloatArray],OrientedBox], extent: float) -> None:
    """Combines corroborating observations of the same compact footprint."""
    box = fit(cloud)
    match = next((c for c in candidates if footprint_overlap(box,fit(c.cloud))>.5
                  and max(fit(np.vstack([c.cloud,cloud])).size[:2])<=extent),None)
    if match is None:
        candidates.append(_Candidate(cloud,{source},{frame:cloud}))
    else:
        match.cloud = np.unique(np.vstack([match.cloud,cloud]),axis=0)
        match.sources.add(source)
        match.observations[frame] = (np.unique(np.vstack([match.observations[frame],cloud]),axis=0)
                                     if frame in match.observations else cloud)


def recover_instances(recon: Reconstruction, tracks: list[Track], clouds: list[FloatArray],
                      fit: Callable[[FloatArray],OrientedBox], config: InstanceConfig,
                      next_id: int) -> list[tuple[list[Track],FloatArray]] | None:
    """Recovers instances from spatially compact observations before assigning broad tracks.

    Args:
        recon: Leveled posed reconstruction.
        tracks: Observations of one category.
        clouds: Fused observed points for those tracks.
        fit: Common oriented-box fitter.
        config: Size, observation and sampling bounds.
        next_id: First globally unused id for corroborated new instances.

    Returns:
        Instance evidence and observed points, or None to retain the ordinary
        fusion if the evidence is ambiguous or would lose an unexplained large object.
    """
    views = [_sample_views(recon,t,config) if len(c)>=config.min_points else []
             for t,c in zip(tracks,clouds)]
    groups,seeds,broad = _seeds(tracks,clouds,fit,config,views)
    if len(seeds)<2 or not broad:
        return None
    if not any(_inside(clouds[i],fit(seed),2*config.voxel).mean()>.2
               for i in broad for seed in seeds):
        return None
    added = _view_candidates(views,broad,seeds,fit,config)
    seeds += [candidate.cloud for candidate in added]
    boxes = [fit(c) for c in seeds]
    # Do not silently split or discard a genuine long conference table.
    if any(np.mean(np.logical_or.reduce([_inside(clouds[i],box,2*config.voxel)
                                        for box in boxes]))<.8 for i in broad):
        return None
    evidence = [[tracks[i] for i in g] for g in groups]
    evidence += [[replace(tracks[min(c.sources)],track_id=next_id+k),
                  *[tracks[i] for i in sorted(c.sources)]] for k,c in enumerate(added)]
    return _own_clouds(tracks,clouds,evidence,boxes,fit,config,broad,[c.cloud for c in added])


def _own_clouds(tracks: list[Track], clouds: list[FloatArray], evidence: list[list[Track]],
                boxes: list[OrientedBox], fit: Callable[[FloatArray],OrientedBox],
                config: InstanceConfig, broad: list[int], extra: list[FloatArray]
                ) -> list[tuple[list[Track],FloatArray]]:
    """Assigns broad observations to bounded instances rather than creating extra slivers."""
    owned = []
    for members,box in zip(evidence,boxes):
        selected = [cloud[_inside(cloud,box,2*config.voxel)]
                    for cloud in [*clouds,*extra]]
        owned.append(np.unique(np.vstack(selected),axis=0))
        for i in broad:
            if _inside(clouds[i],box,2*config.voxel).sum()>=config.min_points \
                    and all(t.track_id != tracks[i].track_id for t in members):
                members.append(tracks[i])
    return list(zip(evidence,split_overlaps(owned,fit)))

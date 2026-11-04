"""Multi-object tracks contribute evidence without becoming cutting boundaries."""

import numpy as np
from synthetic import make_reconstruction, orbit_poses, render

from pivot.geometry.box import fit_box
from pivot.lifting.instances import InstanceConfig, recover_instances
from pivot.lifting.objects import lift_track
from pivot.perception.tracks import Track
from pivot.reconstruction.types import Reconstruction
from pivot.scene.examples import box


def test_recover_missing_desk_from_tracks_switching_between_two_desks():
    boxes = [box((x,y,.375),(.8,1.4,.75))
             for x,y in [(-.6,-.7),(-.6,.7),(.6,.7),(.6,-.7)]]
    poses = orbit_poses(24,radius=4.,height=2.4)
    depths,ids = render(poses,boxes)
    recon = make_reconstruction(poses,depths)
    tracks = [Track(i,'table',{f:ids[f]==i for f in range(len(poses))}) for i in (1,2,3)]
    tracks += [Track(10+i,'table',{f:ids[f]==(4 if (f//2+i)%2 else 3)
                                  for f in range(len(poses))}) for i in (0,1)]
    clouds = [lift_track(recon,t,voxel=.04) for t in tracks]
    result = recover_instances(recon,tracks,clouds,fit_box,InstanceConfig(.04,30,2.,.5),100)
    assert result is not None and len(result)==4
    centers = np.array([fit_box(c).center[:2] for _,c in result])
    for truth in boxes:
        assert np.linalg.norm(centers-truth.center[:2],axis=1).min()<.2
    assert len({group[0].track_id for group,_ in result})==4


def test_unexplained_long_table_is_not_silently_split_or_dropped():
    rng = np.random.default_rng(5)
    clouds = [rng.uniform([-4.,0.,.7],[-2.6,.8,.8],(500,3)),
              rng.uniform([3.,0.,.7],[4.4,.8,.8],(500,3)),
              rng.uniform([-1.5,0.,.7],[1.5,.8,.8],(1000,3))]
    tracks = [Track(i,'table') for i in (1,2,3)]
    recon = Reconstruction([],(1,1),np.empty((0,3,3)),np.empty((0,4,4)),
                           np.empty((0,1,1),dtype=np.float32))
    assert recover_instances(recon,tracks,clouds,fit_box,InstanceConfig(.03,30,2.,.5),10) is None

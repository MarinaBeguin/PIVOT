"""Verified carrying with a short floor departure and arrival when needed."""

from __future__ import annotations

from dataclasses import replace

from pivot.geometry.box import OrientedBox
from pivot.planning.cspace import ConfigurationSpace
from pivot.planning.lift import plan_carried_motion
from pivot.planning.motion import MotionOptions, MotionPlan, MotionStatus, check_stop, plan_motion
from pivot.planning.snap import ranked_free_poses
from pivot.scene.graph import Scene


def _endpoints(scene: Scene, object_id: int, reference: OrientedBox, height: float,
               options: MotionOptions, space: ConfigurationSpace) -> list[MotionPlan]:
    """Finds at most four verified nearby floor departures with a clear lifting column."""
    found = []
    for center,yaw in ranked_free_poses(space,reference.center[:2],reference.yaw,
                                       allow_rotation=False,max_shift=1.):
        check_stop()
        pose = reference.moved(center,yaw)
        probe = plan_carried_motion(scene.moved(object_id,pose),object_id,pose,height,options)
        if probe.status is not MotionStatus.SUCCESS:
            continue
        slide = plan_motion(scene,object_id,pose,options)
        if slide.status is MotionStatus.SUCCESS:
            found.append(slide)
        if len(found)==4:
            break
    return found


def carry_between(scene: Scene, object_id: int, goal: OrientedBox, height: float,
                  options: MotionOptions, space: ConfigurationSpace) -> MotionPlan | None:
    """Tries departure, carry and arrival sequences, with every segment verified.

    Args:
        scene: Complete initial layout.
        object_id: Portable object.
        goal: Free destination on the floor.
        height: Carrying bottom height in meters.
        options: Motion search settings.
        space: Free on-floor configurations of this object.

    Returns:
        A complete motion or no verified sequence within four endpoints per side.
    """
    options = replace(options,max_expansions=min(5000,options.max_expansions))
    departures = _endpoints(scene,object_id,scene[object_id].box,height,options,space)
    arrivals = _endpoints(scene.moved(object_id,goal),object_id,goal,height,options,space)
    for departure in departures:
        for arrival in arrivals:
            carried = plan_carried_motion(scene.moved(object_id,departure.waypoints[-1]),
                object_id,arrival.waypoints[-1],height,options)
            if carried.status is not MotionStatus.SUCCESS:
                continue
            return replace(carried,
                waypoints=(departure.waypoints + carried.waypoints[1:]
                           + list(reversed(arrival.waypoints))[1:]),
                expansions=departure.expansions+carried.expansions+arrival.expansions,
                min_clearance=(min(departure.min_clearance,carried.min_clearance,
                                   arrival.min_clearance)
                               if departure.min_clearance is not None
                               and carried.min_clearance is not None
                               and arrival.min_clearance is not None else None),
                message="Verified floor departure, carry and floor arrival.")
    return None

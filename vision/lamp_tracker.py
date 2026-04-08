"""
vision/lamp_tracker.py — Per-lamp proximity tracking.

Each LampZone has a 2D position in normalised frame space (0-1).

Tracking point selection based on lamp position in frame:
  - Zone in lower half (frame_y >= 0.5): ankle/foot position — best for
    floor and table lamps you walk toward.
  - Zone in upper half (frame_y < 0.5): wrist position — best for wall
    sconces or hanging lights you reach toward.
"""

import logging
from dataclasses import dataclass

from vision.detector import Pose

log = logging.getLogger(__name__)

UPPER_HALF_THRESHOLD = 0.5   # zones with frame_y below this use wrist tracking


@dataclass
class LampZone:
    label: str
    hue_light: str                    # name passed to `openhue set light`
    frame_x: float                    # normalised 0-1
    frame_y: float                    # normalised 0-1
    radius: float                     # proximity threshold (normalised)
    colour_active: tuple[int,int,int] # HSB when someone is near
    colour_idle:   tuple[int,int,int] # HSB when no one is near

    @property
    def uses_wrists(self) -> bool:
        return self.frame_y < UPPER_HALF_THRESHOLD


@dataclass
class LampState:
    zone: LampZone
    occupied: bool = False


def _foot_position(pose: Pose, frame_w: int, frame_h: int) -> tuple[float, float]:
    """
    Normalised floor-level position. Prefers ankle keypoints;
    falls back to bounding-box bottom-centre.
    """
    ankles = []
    for side in ("left", "right"):
        a = pose.kp(f"{side}_ankle")
        if a:
            ankles.append((a.x, a.y))
    if ankles:
        return (
            sum(x for x, _ in ankles) / len(ankles),
            sum(y for _, y in ankles) / len(ankles),
        )
    return (
        (pose.x1 + pose.x2) / 2 / frame_w,
        pose.y2 / frame_h,
    )


def _wrist_position(pose: Pose, frame_w: int, frame_h: int) -> tuple[float, float] | None:
    """
    Normalised position of the highest visible wrist.
    Returns None if no wrists are visible (person can't trigger upper lamps).
    """
    wrists = []
    for side in ("left", "right"):
        w = pose.kp(f"{side}_wrist")
        if w:
            wrists.append((w.x, w.y))
    if not wrists:
        return None
    # Use the wrist with the lowest y (highest in the frame).
    return min(wrists, key=lambda p: p[1])


class LampTracker:
    def __init__(self, zones: list[LampZone]):
        self._states = [LampState(zone=z) for z in zones]
        for z in zones:
            log.info("Zone '%s' → %s tracking (frame_y=%.2f)",
                     z.label, "wrist" if z.uses_wrists else "foot", z.frame_y)

    @property
    def states(self) -> list[LampState]:
        return self._states

    def update(self, poses: list[Pose], frame_w: int, frame_h: int) -> list[LampState]:
        """Recompute which lamps have people near them. Returns updated state list."""
        for ls in self._states:
            ls.occupied = False

        for pose in poses:
            foot = _foot_position(pose, frame_w, frame_h)
            wrist = _wrist_position(pose, frame_w, frame_h)

            for ls in self._states:
                z = ls.zone
                if z.uses_wrists:
                    if wrist is None:
                        continue   # wrists not visible — can't trigger upper lamp
                    px, py = wrist
                else:
                    px, py = foot

                dist = ((px - z.frame_x) ** 2 + (py - z.frame_y) ** 2) ** 0.5
                if dist <= z.radius:
                    ls.occupied = True
                    log.debug("Person near '%s' via %s (dist=%.2f)",
                              z.label, "wrist" if z.uses_wrists else "foot", dist)

        return self._states


def zones_from_config(cfg_zones: list[dict]) -> list[LampZone]:
    """Build LampZone list from config.yaml lamp_zones entries."""
    return [
        LampZone(
            label=entry["label"],
            hue_light=entry["hue_light"],
            frame_x=entry["frame_x"],
            frame_y=entry["frame_y"],
            radius=entry["radius"],
            colour_active=tuple(entry["colour_active"]),
            colour_idle=tuple(entry["colour_idle"]),
        )
        for entry in cfg_zones
    ]

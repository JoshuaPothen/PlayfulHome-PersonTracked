"""
main.py — Entry point.

Usage:
  python3 main.py                # crowd gesture mode
  python3 main.py --mode lamp    # lamp proximity mode
  python3 main.py --headless     # no window (kiosk)

All tuning lives in config.yaml.
Run calibrate.py first before using lamp mode.
"""

import argparse
import logging

import cv2

from utils.config import cfg
from utils.logger import EventLogger
from vision.camera import Camera
from vision.detector import PersonDetector

# ── Logging ───────────────────────────────────────────────────────────────────
_level = getattr(logging, cfg["app"].get("log_level", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_level,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

WINDOW_TITLE = "Person Tracker — Q to quit"


# ── Crowd gesture mode ────────────────────────────────────────────────────────

def run_crowd(headless: bool):
    from state.machine import StateMachine

    event_logger = EventLogger(cfg["app"].get("events_csv", ""))
    detector = PersonDetector(
        model_name=cfg["detection"]["model"],
        confidence=cfg["detection"]["confidence"],
    )
    machine = StateMachine(event_logger=event_logger)
    cam = Camera(device_index=cfg["detection"]["camera_index"])

    try:
        cam.open()
        log.info("Crowd mode%s — Ctrl-C or Q to quit", " (headless)" if headless else "")
        consecutive_none = 0

        while True:
            frame = cam.read()
            if frame is None:
                consecutive_none += 1
                if consecutive_none >= 10:
                    log.error("Camera feed lost — exiting")
                    break
                continue
            consecutive_none = 0

            h, w = frame.shape[:2]
            result = detector.detect(frame)
            machine.update(result, frame_w=w, frame_h=h)

            if not headless:
                annotated = result.annotated_frame.copy()
                cv2.putText(annotated, machine.summary(result, w, h),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(WINDOW_TITLE, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        cam.close()
        event_logger.close()
        if not headless:
            cv2.destroyAllWindows()
        log.info("Done")


# ── Lamp proximity mode ───────────────────────────────────────────────────────

def run_lamp(headless: bool):
    import hue.controller as hue
    from vision.lamp_tracker import LampTracker, zones_from_config

    raw_zones = cfg.get("lamp_zones") or []
    if not raw_zones:
        log.error("No lamp zones configured. Run calibrate.py first.")
        return

    zones = zones_from_config(raw_zones)
    tracker = LampTracker(zones)
    log.info("Lamp mode — %d zone(s): %s", len(zones), [z.label for z in zones])

    # Initialise all lamps to their idle colour.
    for z in zones:
        hue.set_light_color(z.hue_light, *z.colour_idle)

    detector = PersonDetector(
        model_name=cfg["detection"]["model"],
        confidence=cfg["detection"]["confidence"],
    )
    cam = Camera(device_index=cfg["detection"]["camera_index"])

    try:
        cam.open()
        log.info("Lamp mode%s — Ctrl-C or Q to quit", " (headless)" if headless else "")
        consecutive_none = 0

        while True:
            frame = cam.read()
            if frame is None:
                consecutive_none += 1
                if consecutive_none >= 10:
                    log.error("Camera feed lost — exiting")
                    break
                continue
            consecutive_none = 0

            fh, fw = frame.shape[:2]
            result = detector.detect(frame)
            states = tracker.update(result.poses, frame_w=fw, frame_h=fh)

            for ls in states:
                colour = ls.zone.colour_active if ls.occupied else ls.zone.colour_idle
                hue.set_light_color(ls.zone.hue_light, *colour)

            if not headless:
                annotated = result.annotated_frame.copy()
                _draw_lamp_overlay(annotated, states, fw, fh)
                cv2.imshow(WINDOW_TITLE, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        cam.close()
        if not headless:
            cv2.destroyAllWindows()
        log.info("Done")


def _draw_lamp_overlay(frame, states, fw: int, fh: int):
    """Draw lamp zones and their status on the debug frame."""
    overlay = frame.copy()
    COLOURS_BGR = [
        (0, 255, 120), (0, 180, 255), (255, 100, 0),
        (180, 0, 255), (0, 255, 255), (255, 200, 0),
    ]
    for i, ls in enumerate(states):
        z = ls.zone
        cx = int(z.frame_x * fw)
        cy = int(z.frame_y * fh)
        r  = int(z.radius * min(fw, fh))
        bgr = COLOURS_BGR[i % len(COLOURS_BGR)]

        # Zone circle (filled overlay).
        cv2.circle(overlay, (cx, cy), r, bgr, -1)
        # Marker.
        cv2.circle(frame, (cx, cy), 10, bgr, -1)
        cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        # Label with occupied indicator.
        status = "ACTIVE" if ls.occupied else "idle"
        cv2.putText(frame, f"{z.label} [{status}]", (cx + 14, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crowd-reactive light controller")
    parser.add_argument("--mode", choices=["crowd", "lamp"], default="crowd",
                        help="crowd = gesture state machine (default), lamp = proximity zones")
    parser.add_argument("--headless", action="store_true",
                        help="No display window — for kiosk/exhibition deployment")
    args = parser.parse_args()

    headless = args.headless or cfg["app"].get("headless", False)

    if args.mode == "lamp":
        run_lamp(headless)
    else:
        run_crowd(headless)

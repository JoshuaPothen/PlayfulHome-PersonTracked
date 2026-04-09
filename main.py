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

def _person_colour(n: int) -> tuple:
    """
    Return the HSB colour for n people in the frame.
    Looks up config lamp_mode.person_colours; last defined step covers all
    higher counts.
    """
    steps = cfg["lamp_mode"]["person_colours"]
    # steps keys are ints (YAML loads them as int when written as bare numbers)
    sorted_keys = sorted(int(k) for k in steps)
    chosen = sorted_keys[0]
    for k in sorted_keys:
        if n >= k:
            chosen = k
    return tuple(steps[chosen])


def run_lamp(headless: bool):
    import hue.controller as hue
    from vision.lamp_tracker import LampTracker, zones_from_config

    raw_zones = cfg.get("lamp_zones") or []
    if not raw_zones:
        log.error("No lamp zones configured. Run calibrate.py first.")
        return

    lm = cfg["lamp_mode"]
    idle_brightness = lm["idle_brightness"]
    empty_colour    = tuple(lm["empty_colour"])

    zones = zones_from_config(raw_zones)
    tracker = LampTracker(zones)
    log.info("Lamp mode — %d zone(s): %s", len(zones), [z.label for z in zones])

    detector = PersonDetector(
        model_name=cfg["detection"]["model"],
        confidence=cfg["detection"]["confidence"],
    )
    cam = Camera(device_index=cfg["detection"]["camera_index"])

    # Track last applied colour per light to avoid spamming identical commands.
    last_colour: dict[str, tuple] = {}

    def apply(light_name: str, colour: tuple):
        if last_colour.get(light_name) != colour:
            hue.set_light_color(light_name, *colour)
            last_colour[light_name] = colour

    # Start with empty-room state.
    for z in zones:
        apply(z.hue_light, empty_colour)

    try:
        cam.open()
        log.info("Lamp mode%s — Ctrl-C or Q to quit", " (headless)" if headless else "")
        consecutive_none = 0
        prev_count = -1

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
            n = result.person_count
            states = tracker.update(result.poses, frame_w=fw, frame_h=fh)

            if n != prev_count:
                log.info("%d %s in frame → %s", n,
                         "person" if n == 1 else "people",
                         f"hue {_person_colour(n)[0]}°" if n > 0 else "empty")
                prev_count = n

            if n == 0:
                for z in zones:
                    apply(z.hue_light, empty_colour)
            else:
                h_val, s_val, b_val = _person_colour(n)
                for ls in states:
                    if ls.occupied:
                        colour = (h_val, s_val, b_val)
                    else:
                        colour = (h_val, s_val, idle_brightness)
                    apply(ls.zone.hue_light, colour)

            if not headless:
                annotated = result.annotated_frame.copy()
                _draw_lamp_overlay(annotated, states, fw, fh, n)
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


def _draw_lamp_overlay(frame, states, fw: int, fh: int, person_count: int = 0):
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

    # Person count + active colour step.
    if person_count == 0:
        label = "0 people — empty"
    else:
        h_val = _person_colour(person_count)[0]
        label = f"{person_count} {'person' if person_count == 1 else 'people'} — hue {h_val}\u00b0"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


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

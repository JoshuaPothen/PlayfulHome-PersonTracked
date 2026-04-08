"""
calibrate.py — Interactive lamp zone calibration.

Opens the camera and lets you click on each lamp in the frame.
For each click you enter the matching Hue light name in the terminal.
When done, zones are saved to config.yaml under `lamp_zones`.

Controls (in the camera window):
  Left-click     — place a lamp marker at the cursor
  U              — undo the last placed marker
  S / Enter      — save and quit
  Q / Escape     — quit without saving
"""

import sys
import pathlib
import logging

import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"

# Visual style
MARKER_RADIUS  = 12
ZONE_ALPHA     = 0.25   # overlay transparency for zone circles
COLOURS = [  # BGR, one per lamp
    (0, 255, 120),
    (0, 180, 255),
    (255, 100, 0),
    (180, 0, 255),
    (0, 255, 255),
    (255, 200, 0),
]

DEFAULT_COLOUR_ACTIVE = [120, 100, 90]
DEFAULT_COLOUR_IDLE   = [30,  20,  40]
DEFAULT_RADIUS        = 0.18   # normalised proximity zone


def pick_colour(idx: int):
    return COLOURS[idx % len(COLOURS)]


class Calibrator:
    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.markers: list[dict] = []   # {px, py, label, hue_light}
        self._pending: tuple[int,int] | None = None  # click awaiting input

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._pending = (x, y)
            log.info("Lamp placed at pixel (%d, %d) — check terminal", x, y)

    def has_pending(self) -> bool:
        return self._pending is not None

    def collect_pending(self) -> tuple[int,int] | None:
        p = self._pending
        self._pending = None
        return p

    def add_marker(self, px: int, py: int, label: str, hue_light: str):
        self.markers.append({
            "label":          label,
            "hue_light":      hue_light,
            "frame_x":        round(px / self.frame_w, 3),
            "frame_y":        round(py / self.frame_h, 3),
            "radius":         DEFAULT_RADIUS,
            "colour_active":  DEFAULT_COLOUR_ACTIVE,
            "colour_idle":    DEFAULT_COLOUR_IDLE,
        })
        log.info("Added zone '%s' → light '%s'  (%.3f, %.3f)",
                 label, hue_light, px / self.frame_w, py / self.frame_h)

    def undo(self):
        if self.markers:
            removed = self.markers.pop()
            log.info("Removed zone '%s'", removed["label"])

    def draw(self, frame):
        overlay = frame.copy()
        h, w = frame.shape[:2]

        for i, m in enumerate(self.markers):
            bgr = pick_colour(i)
            cx = int(m["frame_x"] * w)
            cy = int(m["frame_y"] * h)
            r  = int(m["radius"] * min(w, h))

            # Filled circle overlay for zone.
            cv2.circle(overlay, (cx, cy), r, bgr, -1)
            # Marker dot + cross.
            cv2.circle(frame, (cx, cy), MARKER_RADIUS, bgr, -1)
            cv2.drawMarker(frame, (cx, cy), (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            # Label.
            cv2.putText(frame, m["label"], (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

        # Blend zone circles.
        cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1 - ZONE_ALPHA, 0, frame)

        # HUD.
        lines = [
            f"Lamps placed: {len(self.markers)}",
            "Left-click = place lamp",
            "U = undo   S = save   Q = quit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame


def prompt_lamp_info(index: int) -> tuple[str, str] | None:
    """Ask for label and Hue light name in the terminal. Returns None to cancel."""
    print(f"\n── Lamp {index + 1} ──")
    label = input("  Label (e.g. 'Corner Lamp'): ").strip()
    if not label:
        print("  Cancelled.")
        return None
    hue_light = input("  Hue light name (run `openhue get lights` to list): ").strip()
    if not hue_light:
        print("  Cancelled.")
        return None
    return label, hue_light


def save_zones(markers: list[dict]):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["lamp_zones"] = markers
    CONFIG_PATH.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    log.info("Saved %d lamp zone(s) to %s", len(markers), CONFIG_PATH)


def run():
    from utils.config import cfg
    cam_index = cfg["detection"]["camera_index"]

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        log.error("Could not open camera %d", cam_index)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cal = Calibrator(frame_w=w, frame_h=h)

    cv2.namedWindow("Calibrate — click on each lamp")
    cv2.setMouseCallback("Calibrate — click on each lamp", cal.on_mouse)

    print("\n=== Lamp Calibration ===")
    print("Click on each lamp in the camera window, then fill in the terminal prompts.")
    print("Press S to save when done.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            log.error("Camera read failed")
            break

        # Handle pending click — must pause camera loop for terminal input.
        if cal.has_pending():
            px, py = cal.collect_pending()
            info = prompt_lamp_info(len(cal.markers))
            if info:
                cal.add_marker(px, py, *info)

        display = cal.draw(frame.copy())
        cv2.imshow("Calibrate — click on each lamp", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # Q or Escape
            print("Quit without saving.")
            break
        if key == ord("u"):
            cal.undo()
        if key in (ord("s"), 13):   # S or Enter
            if cal.markers:
                save_zones(cal.markers)
                print(f"\nSaved {len(cal.markers)} zone(s) to config.yaml")
                print("Edit radius / colour_active / colour_idle in config.yaml to fine-tune.")
            else:
                print("No zones to save.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()

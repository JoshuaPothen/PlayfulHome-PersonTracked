"""
test_hue.py — Manual smoke test for hue/controller.py.

Cycles through a handful of colours so you can visually confirm the
wrapper is calling the real lights. Run with:

    python test_hue.py

The script sleeps briefly between commands so you can see each change.
Pass --dry-run to print the commands without executing them (useful when
the Hue Bridge is unreachable).
"""

import argparse
import logging
import subprocess
import sys
import time
import unittest.mock

# Configure logging so controller.py debug output is visible.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(name)s: %(message)s",
)

import hue.controller as ctrl

COLOURS = [
    # (label,       hue,  sat,  bri)
    ("Red",          0,   100,  80),
    ("Orange",      30,   100,  80),
    ("Yellow",      60,   100,  80),
    ("Green",      120,   100,  70),
    ("Cyan",       180,   100,  70),
    ("Blue",       240,   100,  80),
    ("Purple",     270,   100,  80),
    ("White",        0,     0, 100),
]

STEP_DELAY = 1.5  # seconds between changes


def run_live():
    print("\n=== Hue Wrapper Smoke Test (LIVE) ===\n")

    print("[1] Turning lights ON…")
    ok = ctrl.lights_on()
    print(f"    lights_on() → {'OK' if ok else 'FAILED'}")
    time.sleep(STEP_DELAY)

    print("\n[2] Cycling through colours…")
    for label, h, s, b in COLOURS:
        print(f"    set_color({h:3d}°, {s}%, {b}%) — {label}")
        ok = ctrl.set_color(h, s, b)
        print(f"    → {'OK' if ok else 'FAILED'}")
        time.sleep(STEP_DELAY)

    print("\n[3] Setting brightness to 30%…")
    ok = ctrl.set_brightness(30)
    print(f"    set_brightness(30) → {'OK' if ok else 'FAILED'}")
    time.sleep(STEP_DELAY)

    print("\n[4] Warm white (500 mirek)…")
    ok = ctrl.set_color_temperature(500)
    print(f"    set_color_temperature(500) → {'OK' if ok else 'FAILED'}")
    time.sleep(STEP_DELAY)

    print("\n[5] Turning lights OFF…")
    ok = ctrl.lights_off()
    print(f"    lights_off() → {'OK' if ok else 'FAILED'}")

    print("\n=== Done ===\n")


def run_dry():
    """Print what commands would be sent without hitting the bridge."""
    print("\n=== Hue Wrapper Smoke Test (DRY RUN) ===\n")

    captured = []

    original_run = subprocess.run

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        print("  CMD:", " ".join(str(c) for c in cmd))

        FAKE_LIGHTS = '[{"id":"fake-id-1","name":"Light 1"},{"id":"fake-id-2","name":"Light 2"}]'

        class FakeResult:
            returncode = 0
            stdout = FAKE_LIGHTS
            stderr = ""

        return FakeResult()

    with unittest.mock.patch("subprocess.run", side_effect=fake_run):
        ctrl.lights_on()
        ctrl.set_color(0, 100, 80)    # Red
        ctrl.set_color(120, 100, 70)  # Green
        ctrl.set_color(240, 100, 80)  # Blue
        ctrl.set_brightness(50)
        ctrl.set_color_temperature(300)
        ctrl.lights_off()

    print(f"\n{len(captured)} commands would be issued.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test the Hue wrapper.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    if args.dry_run:
        run_dry()
    else:
        run_live()

"""
hue/controller.py — OpenHue CLI subprocess wrapper.

When DEFAULT_ROOM is set, every command uses `openhue set room <name>`
(single round-trip, no pre-fetch).  When it is None the wrapper falls
back to fetching all light IDs first.

Public API
----------
set_color(hue, saturation, brightness)   HSB 0-360 / 0-100 / 0-100
lights_on()
lights_off()
set_brightness(brightness)               0-100
set_color_temperature(mirek)             153 (cool) – 500 (warm)
"""

import colorsys
import json
import logging
import subprocess
from typing import Optional

log = logging.getLogger(__name__)

# Set this to your gallery room name, e.g. "Gallery" or "Living Room".
# When set, every command becomes `openhue set room <name> …` (one call).
# When None, the wrapper fetches all light IDs each time.
DEFAULT_ROOM: Optional[str] = "Living room"

OPENHUE_BIN = "openhue"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(args: list[str]) -> bool:
    """Run an openhue command. Returns True on success, False on failure."""
    cmd = [OPENHUE_BIN] + args
    log.debug("CMD: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("openhue error (exit %d): %s", result.returncode, result.stderr.strip())
        return False
    return True


def _get_light_ids() -> list[str]:
    """Return IDs for every light. Returns [] on failure."""
    result = subprocess.run(
        [OPENHUE_BIN, "get", "lights", "--json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error("Could not fetch lights: %s", result.stderr.strip())
        return []
    try:
        lights = json.loads(result.stdout)
        return [l["id"] for l in lights if "id" in l]
    except (json.JSONDecodeError, TypeError) as exc:
        log.error("Failed to parse lights JSON: %s", exc)
        return []


def _hsb_to_rgb_hex(hue: float, saturation: float, brightness: float) -> str:
    """Convert HSB (hue 0-360, sat 0-100, bri 0-100) to '#RRGGBB'."""
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, brightness / 100.0)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def _build_set_cmd(room: Optional[str], extra_flags: list[str]) -> Optional[list[str]]:
    """
    Build a complete `set light …` or `set room …` command.
    Returns None if no targets could be determined.
    """
    target_room = room or DEFAULT_ROOM
    if target_room:
        # Single command — no pre-fetch needed.
        return ["set", "room", target_room] + extra_flags
    # Fall back: fetch all light IDs.
    ids = _get_light_ids()
    if not ids:
        log.warning("No lights found — is the bridge reachable?")
        return None
    return ["set", "light"] + ids + extra_flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_color(
    hue: float,
    saturation: float,
    brightness: float,
    room: Optional[str] = None,
) -> bool:
    """Set colour (HSB) and turn lights on."""
    rgb = _hsb_to_rgb_hex(hue, saturation, brightness)
    log.info("set_color hue=%.0f sat=%.0f bri=%.0f → %s", hue, saturation, brightness, rgb)
    cmd = _build_set_cmd(room, ["--on", "--rgb", rgb, "--brightness", str(brightness)])
    return _run(cmd) if cmd else False


def lights_on(room: Optional[str] = None) -> bool:
    """Turn lights on."""
    log.info("lights_on")
    cmd = _build_set_cmd(room, ["--on"])
    return _run(cmd) if cmd else False


def lights_off(room: Optional[str] = None) -> bool:
    """Turn lights off."""
    log.info("lights_off")
    cmd = _build_set_cmd(room, ["--off"])
    return _run(cmd) if cmd else False


def set_brightness(brightness: float, room: Optional[str] = None) -> bool:
    """Set brightness (0-100) without changing colour."""
    log.info("set_brightness %.0f", brightness)
    cmd = _build_set_cmd(room, ["--on", "--brightness", str(brightness)])
    return _run(cmd) if cmd else False


def set_color_temperature(mirek: int, room: Optional[str] = None) -> bool:
    """Set colour temperature in Mirek (153 = cool white, 500 = warm white)."""
    mirek = max(153, min(500, mirek))
    log.info("set_color_temperature %d mirek", mirek)
    cmd = _build_set_cmd(room, ["--on", "--temperature", str(mirek)])
    return _run(cmd) if cmd else False

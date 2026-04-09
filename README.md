# PlayfulHome — Person Tracked Light Controller

A local, real-time system that watches a room through a camera and uses crowd pose detection to control Philips Hue lights. No cloud services, no apps — just a camera, a Hue bridge, and a Python process.

Two independent modes:

- **Crowd mode** — detects crowd gestures (squat, hands up, huddle, stillness) and maps them to room-wide light states
- **Lamp mode** — maps physical lamps to zones in the camera frame; walking toward or reaching for a lamp triggers that specific light, with colour shifting based on how many people are in the room

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  Camera      │───▶│  PersonDetector  │───▶│ StateMachine  │  │
│  │  (OpenCV)    │    │  (YOLOv8-Pose)   │    │  or           │  │
│  └──────────────┘    └──────────────────┘    │ LampTracker   │  │
│                               │              └───────┬───────┘  │
│                               │                      │          │
│                        Pose keypoints                │          │
│                        (17 joints, norm.)            ▼          │
│                                             ┌────────────────┐  │
│                                             │ hue/controller │  │
│                                             │ (subprocess)   │  │
│                                             └───────┬────────┘  │
└─────────────────────────────────────────────────────┼───────────┘
                                                      │
                                          openhue CLI (local)
                                                      │
                                          Philips Hue Bridge
                                                      │
                                              Hue Lights
```

### Data flow (per frame)

```
Camera frame
    │
    ▼
YOLOv8-Pose inference
    │
    ├─▶ Bounding boxes (pixel coords)
    └─▶ 17 keypoints per person (normalised 0–1, y increases downward)
              │
              ├── CROWD MODE
              │       │
              │   vision/gestures.py
              │       ├─ is_squatting()      hip_y / knee_y ratio
              │       ├─ is_hands_raised()   wrist_y < shoulder_y
              │       ├─ huddle_direction()  centre-of-mass x position
              │       └─ mean_movement()     frame-to-frame displacement
              │               │
              │       state/machine.py
              │               ├─ Evaluate all conditions
              │               ├─ Pick highest-priority active state
              │               ├─ Enforce per-state hold time (cooldown)
              │               └─ Apply light colour via hue/controller.py
              │
              └── LAMP MODE
                      │
                  vision/lamp_tracker.py
                      ├─ Lower zones → foot/ankle position
                      ├─ Upper zones → highest wrist position
                      ├─ Proximity check per zone (Euclidean, normalised)
                      └─ Apply per-light colour via hue/controller.py
                              (colour determined by person count in frame)
```

### Module map

| Path | Responsibility |
|---|---|
| `main.py` | Entry point; CLI args; runs crowd or lamp loop |
| `config.yaml` | All tuning — no code changes needed |
| `calibrate.py` | Interactive lamp zone calibration |
| `hue/controller.py` | OpenHue CLI subprocess wrapper |
| `vision/camera.py` | OpenCV webcam feed with auto-recovery |
| `vision/detector.py` | YOLOv8-Pose inference; `Pose` + `Keypoint` dataclasses |
| `vision/gestures.py` | Per-person and crowd gesture functions |
| `vision/lamp_tracker.py` | Lamp zone proximity logic |
| `state/machine.py` | Crowd gesture state machine with priority + cooldowns |
| `utils/config.py` | Loads `config.yaml` once at import |
| `utils/logger.py` | CSV event logger for state transitions |

---

## Requirements

- Python 3.11+
- A Philips Hue bridge on the same local network
- [OpenHue CLI](https://www.openhue.io/cli) installed and configured
- A webcam visible to the machine

### Python dependencies

```bash
pip install -r requirements.txt
```

> YOLOv8 model weights (`yolov8s-pose.pt`, ~22 MB) are downloaded automatically on first run.

---

## Setup

### 1. Install and configure OpenHue CLI

Follow the [OpenHue setup guide](https://www.openhue.io/cli/setup) to pair the CLI with your bridge. Verify it works:

```bash
openhue get lights
openhue get rooms
```

Note the exact light names and room name — these are case-sensitive and must match `config.yaml`.

### 2. Clone and install dependencies

```bash
git clone <repo-url>
cd PlayfulHome-PersonTracked
pip install -r requirements.txt
```

### 3. Configure `config.yaml`

Set your room name and camera index at minimum:

```yaml
hue:
  room: "Living room"   # exact name from `openhue get rooms`

detection:
  camera_index: 0       # 0 = default webcam; increment if wrong camera opens
```

### 4. Test the Hue connection

```bash
python3 test_hue.py --dry-run   # verify command structure without hitting bridge
python3 test_hue.py             # cycles through colours on real lights
```

---

## Running

### Crowd gesture mode (default)

```bash
python3 main.py
```

### Lamp proximity mode

Run calibration first (see below), then:

```bash
python3 main.py --mode lamp
```

### Headless / kiosk mode

```bash
python3 main.py --headless
python3 main.py --mode lamp --headless
```

Or set `headless: true` in `config.yaml` to make it permanent.

Press **Q** in the debug window (or **Ctrl-C** in the terminal) to quit.

---

## Crowd Mode — Interaction Patterns

The system detects six states. When multiple conditions are true simultaneously, the highest-priority state wins. Each state has a minimum hold time before transitioning, preventing flickering.

### State priority (highest → lowest)

| Priority | State | Trigger | Light |
|---|---|---|---|
| 1 | **HANDS UP** | ≥50% of people have a wrist above shoulder | Full white — 100% brightness |
| 2 | **SQUAT** | ≥50% of people squatting | Vivid purple |
| 3 | **HUDDLE RIGHT** | Crowd centre-of-mass past 60% of frame width | Warm orange |
| 3 | **HUDDLE LEFT** | Crowd centre-of-mass below 40% of frame width | Cool blue |
| 4 | **STILLNESS** | No significant movement for 4 seconds | Dim warm — subtle ambient |
| 5 | **IDLE** | Default | Soft warm white |

### Gesture reference

**Squat**
- Hip y-coordinate drops to ≥85% of knee y-coordinate (normalised frame height)
- Proportion-based: needs 50% of detected people to trigger, so a solo performer can trigger it alone but one person in a group of 10 cannot
- Cooldown: 5 seconds in state before transitioning out

**Hands up**
- Either wrist must be above the shoulder on the same side
- Set `hands_raised_min_wrists: 2` in config to require both hands

**Huddle left / right**
- Calculates the mean x-position of all bounding box centres
- Triggers when COM drifts past the 40%/60% threshold
- Requires at least 2 people (`huddle_min_people`)

**Stillness**
- Measures average frame-to-frame displacement of bounding box centres (normalised)
- Triggers after movement stays below `still_movement_thresh` for `still_duration` seconds
- Resets immediately on movement

### Tuning for your space

All thresholds live in `config.yaml` — no code edits required:

```yaml
gestures:
  squat_hip_knee_ratio: 0.85    # raise → harder to trigger (stricter squat)
  hands_raised_min_wrists: 1    # 2 = both hands must be raised
  huddle_left_threshold: 0.40   # lower = harder to trigger huddle left
  huddle_right_threshold: 0.60  # higher = harder to trigger huddle right
  huddle_min_people: 2
  still_movement_thresh: 0.015  # lower = must be more still to trigger
  still_duration: 4.0           # seconds of stillness before triggering

states:
  squat_trigger_ratio: 0.50     # fraction of crowd that must be squatting
  hands_up_ratio: 0.50
  min_people: 1                 # ignore gestures below this headcount
```

---

## Lamp Mode — Interaction Patterns

Each physical lamp in the room is mapped to a zone in the camera frame. The system independently tracks each lamp zone.

### Proximity logic

- **Lower-half zones** (`frame_y ≥ 0.5`): triggered by foot/ankle position — best for floor lamps and table lamps you walk toward
- **Upper-half zones** (`frame_y < 0.5`): triggered by wrist position — best for wall sconces or hanging lights you reach toward

This means you walk up to a floor lamp to activate it, but you raise your hand toward a wall light.

### Colour by crowd size

The colour of all active lamps shifts with the total number of people detected in frame:

| People in frame | Active lamp (nearby) | Idle lamp (not nearby) |
|---|---|---|
| 0 | dim warm amber | dim warm amber |
| 1 | green (120°) @ 90% | green (120°) @ 20% |
| 2 | blue (210°) @ 90% | blue (210°) @ 20% |
| 3 | purple (280°) @ 90% | purple (280°) @ 20% |
| 4+ | orange (30°) @ 90% | orange (30°) @ 20% |

Idle lamps always show the same hue as the current crowd count — dimly — giving a subtle ambient reading of the room state even without interaction.

Colour steps are fully configurable in `config.yaml`:

```yaml
lamp_mode:
  idle_brightness: 20
  empty_colour: [30, 20, 15]
  person_colours:
    1: [120, 100, 90]
    2: [210, 100, 90]
    3: [280, 100, 90]
    4: [30,  100, 90]
    5: [0,   100, 90]   # add more steps as needed
```

---

## Lamp Calibration Routine

Run once before using lamp mode. You will click on each lamp in the camera view and provide its Hue light name.

```bash
python3 calibrate.py
```

### Step-by-step

1. The camera feed opens in a window
2. **Click** on a lamp in the frame — a coloured marker appears
3. Switch to the terminal — enter a label (e.g. `Corner Lamp`) and the exact Hue light name (from `openhue get lights`)
4. Repeat for each lamp
5. Press **S** (or Enter) to save zones to `config.yaml`
6. Press **U** to undo the last marker if you misclick
7. Press **Q** to quit without saving

After saving, zones appear in `config.yaml` under `lamp_zones`. You can fine-tune `radius`, `colour_active`, and `colour_idle` by hand — no re-calibration needed.

```yaml
lamp_zones:
  - label: "Corner Lamp"
    hue_light: "Hue Go"     # must match `openhue get lights` exactly
    frame_x: 0.292          # normalised x position in frame (0 = left, 1 = right)
    frame_y: 0.621          # normalised y position (0 = top, 1 = bottom)
    radius: 0.18            # zone size — increase if hard to trigger
    colour_active: [120, 100, 90]
    colour_idle:   [30,  20,  40]
```

### Tips

- Position the camera with all lamps in view before calibrating
- Click on the lamp itself (the bulb / shade), not the wall behind it
- Zones in the top half of the frame automatically use wrist tracking — click where the light is, not where you stand
- If a zone is hard to trigger, increase its `radius` in `config.yaml`
- Run `openhue get lights` to get exact light names before calibrating

---

## Project Structure

```
PlayfulHome-PersonTracked/
├── main.py                  # Entry point — crowd or lamp mode
├── calibrate.py             # Interactive lamp zone calibration
├── test_hue.py              # Hue wrapper smoke test
├── config.yaml              # All tuning — thresholds, colours, zones
├── .gitignore
│
├── hue/
│   └── controller.py        # OpenHue CLI subprocess wrapper
│                            # set_color(), lights_on/off(), set_light_color()
│
├── vision/
│   ├── camera.py            # OpenCV feed with auto-recovery
│   ├── detector.py          # YOLOv8-Pose inference → Pose + Keypoint dataclasses
│   ├── gestures.py          # is_squatting(), is_hands_raised(), huddle_direction()
│   └── lamp_tracker.py      # LampZone proximity tracking
│
├── state/
│   └── machine.py           # Crowd gesture state machine (6 states, priority + cooldowns)
│
└── utils/
    ├── config.py            # Loads config.yaml once at import
    └── logger.py            # CSV event logger (events.csv)
```

---

## Configuration Reference

Full `config.yaml` with all fields explained:

```yaml
hue:
  room: "Living room"        # Room name from `openhue get rooms`. Controls all
                             # lights in the room as a single command.

detection:
  model: "yolov8s-pose.pt"  # yolov8n-pose.pt = faster/less accurate
                             # yolov8s-pose.pt = balanced (recommended)
                             # yolov8m-pose.pt = more accurate, slower
  confidence: 0.4            # Detection confidence threshold (0–1)
  camera_index: 0            # Webcam index; increment if wrong camera opens

gestures:
  squat_hip_knee_ratio: 0.85 # hip_y / knee_y ≥ this → squatting
  hands_raised_min_wrists: 1 # 1 = either hand; 2 = both hands required
  huddle_left_threshold: 0.40
  huddle_right_threshold: 0.60
  huddle_min_people: 2
  still_movement_thresh: 0.015
  still_duration: 4.0

states:
  squat_trigger_ratio: 0.50  # fraction of people required (proportion-based)
  hands_up_ratio: 0.50
  min_people: 1              # gestures ignored below this headcount
  idle:
    colour: [30, 20, 70]     # HSB: hue 0–360, saturation 0–100, brightness 0–100
    min_hold: 3.0            # seconds before transitioning out of this state
  stillness:
    colour: [30, 30, 20]
    min_hold: 8.0
  huddle_left:
    colour: [210, 80, 80]
    min_hold: 4.0
  huddle_right:
    colour: [30, 90, 80]
    min_hold: 4.0
  squat_active:
    colour: [280, 100, 90]
    min_hold: 5.0
  hands_up:
    colour: [0, 0, 100]
    min_hold: 4.0

app:
  headless: false            # true = no OpenCV window (kiosk/exhibition)
  log_level: INFO            # DEBUG streams raw keypoint data per frame
  events_csv: "events.csv"  # State transition log; set to "" to disable

lamp_mode:
  idle_brightness: 20
  empty_colour: [30, 20, 15]
  person_colours:
    1: [120, 100, 90]
    2: [210, 100, 90]
    3: [280, 100, 90]
    4: [30,  100, 90]

lamp_zones:                  # Populated by calibrate.py
  - label: "Corner Lamp"
    hue_light: "Hue Go"
    frame_x: 0.292
    frame_y: 0.621
    radius: 0.18
    colour_active: [120, 100, 90]
    colour_idle:   [30,  20,  40]
```

---

## Event Logging

When `events_csv` is set in `config.yaml`, every state transition in crowd mode is appended to a CSV file:

```
timestamp,from_state,to_state,person_count,squat_count,hands_count
2026-04-08T19:32:11,IDLE,SQUAT,5,3,0
2026-04-08T19:32:18,SQUAT,HANDS UP,5,1,4
2026-04-08T19:32:24,HANDS UP,IDLE,3,0,0
```

Useful for reviewing what interactions happened during a session. Disable with `events_csv: ""`.

---

## Deployment Tips

### Long-running exhibition sessions

- Use `--headless` to avoid the overhead of rendering the debug window
- Set `log_level: WARNING` in `config.yaml` to reduce console output
- Run inside `screen` or `tmux` so it survives terminal disconnection:
  ```bash
  screen -S tracker
  python3 main.py --headless
  # Ctrl-A D to detach
  ```

### Camera placement

- Mount the camera high and angled down slightly for the best full-body keypoint detection — the model needs to see hips and knees to detect squats reliably
- Avoid backlighting; the camera should face away from windows
- For lamp mode, ensure all lamps are visible in the frame before calibrating

### Model selection

| Model | Speed | Accuracy | Recommended for |
|---|---|---|---|
| `yolov8n-pose.pt` | Fastest | Lower | Low-powered machines |
| `yolov8s-pose.pt` | Balanced | Good | Most setups (default) |
| `yolov8m-pose.pt` | Slower | Higher | High-accuracy needs, powerful machine |

Change via `detection.model` in `config.yaml`.

### Auto-start on boot (macOS)

Create a launch agent at `~/Library/LaunchAgents/com.playfulmome.tracker.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.playfulmhome.tracker</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>/path/to/PlayfulHome-PersonTracked/main.py</string>
    <string>--headless</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
</dict>
</plist>
```

Load it with:
```bash
launchctl load ~/Library/LaunchAgents/com.playfulmhome.tracker.plist
```

---

## Troubleshooting

**Lights not responding**
- Run `openhue get lights` — if this fails, the bridge is unreachable
- Confirm the machine and bridge are on the same network
- Check the room name in `config.yaml` matches exactly (case-sensitive)

**Wrong camera opens**
- Increment `camera_index` in `config.yaml` (try 1, 2, …)
- On macOS, grant camera permission to Terminal in System Settings → Privacy

**Gestures not triggering**
- Check the debug window overlay — it shows current state, person count, squat %, and hands %
- Lower `squat_trigger_ratio` / `hands_up_ratio` to make triggers easier
- Ensure the camera can see hips and knees (squat) or shoulders and wrists (hands up)
- Increase `confidence` if ghost detections are causing false triggers; decrease it if real people aren't being detected

**Lamp zones not triggering**
- Increase `radius` for the zone in `config.yaml`
- Check the debug overlay — zones are drawn with ACTIVE/idle labels
- For upper-half zones, the wrist must be visible to the model — arms must be clearly raised

**Jittery light changes**
- Increase `min_hold` for the relevant state
- Increase `still_duration` if STILLNESS triggers too easily
- Reduce `log_level` to `WARNING` to verify it's not a logging bottleneck

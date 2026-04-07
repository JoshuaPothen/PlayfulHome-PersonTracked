# Crowd-Reactive Light Control — Build Plan
> Camera vision → pose detection → Philips Hue  
> Single front-facing camera · Medium gallery (10–30 people) · Local machine

---

## Phase 1 — OpenHue CLI Subprocess Wrapper

**Goal:** Wrap the existing OpenHue CLI so the rest of the codebase can trigger lights with a simple function call. No camera yet.

> OpenHue CLI is already configured and working. Python just needs to call it via subprocess.

**Tasks:**
- Confirm which OpenHue CLI commands are used (e.g. `openhue set light ...`)
- Write a `hue/controller.py` wrapper with helper functions: `set_color(hue, saturation, brightness)`, `lights_on()`, `lights_off()`
- Each helper runs the appropriate CLI command via `subprocess.run()`
- Capture stderr and handle failures gracefully (log and continue, don't crash)
- Write a quick test script that cycles through a few colours to confirm the wrapper works

**Stack:** Python `subprocess`, existing OpenHue CLI  
**Milestone:** `controller.set_color(...)` triggers a real light change with one function call.

---

## Phase 2 — Camera Feed & Person Detection

**Goal:** See people in the frame. No pose analysis yet.

**Tasks:**
- Open webcam feed with OpenCV
- Run YOLOv8 (nano model) to detect people in each frame
- Draw bounding boxes around detected people
- Display a live debug window showing the feed + detections
- Log detected person count to the console in real time

**Stack:** Python, OpenCV, `ultralytics` (YOLOv8)  
**Milestone:** Console prints "3 people detected" while showing a live annotated camera feed.

---

## Phase 3 — Pose Keypoint Extraction

**Goal:** Know what bodies are doing, not just where they are.

**Tasks:**
- Switch from YOLOv8 object detection to YOLOv8-Pose (or MediaPipe Pose)
- Extract keypoints per detected person: hips, knees, shoulders, wrists, ankles
- Normalise keypoint coordinates relative to frame size
- Display skeleton overlay on the live debug feed
- Log raw keypoint data per person per frame

**Stack:** `ultralytics` YOLOv8-Pose or `mediapipe`  
**Milestone:** Skeleton lines drawn on every detected person in the live feed.

---

## Phase 4 — Single Gesture Detection

**Goal:** Detect one gesture reliably. Validate the pipeline end-to-end.

**Gesture to implement:** Squat  
Detection logic: hip keypoint y-position drops below knee threshold for a person → that person is squatting.

**Tasks:**
- Write `is_squatting(keypoints)` function using hip/knee y-ratio
- Count how many detected people are currently squatting
- When count ≥ threshold (e.g. 3 people), trigger a Hue colour change
- Add a cooldown timer (e.g. 5 seconds) so lights don't flicker
- Log gesture state changes to console

**Milestone:** 3+ people squat → lights change colour. Stand up → lights return.

---

## Phase 5 — State Machine & Multiple Interactions

**Goal:** Support multiple distinct interactions with clean state management.

**Interactions to implement:**
- **Squat** — N people squat → colour change (from Phase 4)
- **Huddle left/right** — centre-of-mass of all bounding boxes shifts past a threshold → lights shift colour temperature
- **Hands raised** — wrist keypoints above head for N people → lights brighten to full
- **Stillness** — no significant movement detected for X seconds → lights dim slowly

**Tasks:**
- Build a `StateMachine` class with named states (e.g. `IDLE`, `SQUAT_ACTIVE`, `HUDDLE_LEFT`, `HANDS_UP`)
- Each state maps to a Hue scene or colour config
- Define entry conditions and exit conditions per state
- Handle state priority if multiple conditions are true simultaneously
- Add per-state cooldowns to prevent rapid flickering

**Milestone:** All four interactions work independently and transition cleanly between light states.

---

## Phase 6 — Robustness & Gallery-Ready Polish

**Goal:** Reliable enough to run unattended during an exhibition.

**Tasks:**
- Tune detection thresholds for the actual space and lighting conditions
- Handle occlusion gracefully — use proportion-based logic (e.g. "60% of detected people" not hard count)
- Auto-recover if camera feed drops or Bridge connection is lost
- Add a headless mode (no debug window) for kiosk deployment
- Write a simple config file (`config.yaml`) for tuning thresholds, cooldowns, and Hue scene IDs without touching code
- Optional: log interaction events to a CSV for post-show data review

**Milestone:** System runs for a 4-hour session without intervention. Interactions feel responsive and intentional to participants.

---

## Interaction Reference

| Interaction | Detection Signal | Difficulty |
|---|---|---|
| Squat | Hip/knee keypoint ratio | Medium |
| Huddle to one side | Bounding box centre-of-mass | Easy |
| Hands raised | Wrist above head keypoints | Medium |
| Stillness / freeze | Frame-to-frame movement delta | Easy |
| Form a line | Horizontal spread of bounding boxes | Medium |
| Face same direction | Shoulder orientation (left/right asymmetry) | Hard |
| Sit on floor | Hip keypoint drops very low | Medium |

---

## Project Structure (Target)

```
project/
├── main.py                  # Entry point
├── config.yaml              # Thresholds, cooldowns, Hue config
├── hue/
│   ├── bridge.py            # Bridge auth + connection
│   └── controller.py        # Light control helpers
├── vision/
│   ├── camera.py            # OpenCV feed
│   ├── detector.py          # YOLOv8-Pose inference
│   └── gestures.py          # Gesture logic functions
├── state/
│   └── machine.py           # State machine
└── utils/
    └── logger.py            # Event logging
```

---

## Dependencies

```
opencv-python
ultralytics        # YOLOv8 + YOLOv8-Pose
mediapipe          # Alternative pose option
pyyaml             # Config file parsing
numpy              # Keypoint math
```

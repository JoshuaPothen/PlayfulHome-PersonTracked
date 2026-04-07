"""
vision/detector.py — YOLOv8-Pose person detector.

Switches from plain detection to pose estimation so downstream gesture
logic can read named joint positions.

COCO 17-keypoint indices used by YOLOv8-Pose:
  0 nose        1 left_eye     2 right_eye    3 left_ear     4 right_ear
  5 left_shoulder  6 right_shoulder  7 left_elbow  8 right_elbow
  9 left_wrist  10 right_wrist  11 left_hip   12 right_hip
  13 left_knee  14 right_knee  15 left_ankle  16 right_ankle
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)

# Subset of COCO keypoints used by gesture logic.
KEYPOINT_NAMES: dict[str, int] = {
    "left_shoulder":  5,
    "right_shoulder": 6,
    "left_elbow":     7,
    "right_elbow":    8,
    "left_wrist":     9,
    "right_wrist":   10,
    "left_hip":      11,
    "right_hip":     12,
    "left_knee":     13,
    "right_knee":    14,
    "left_ankle":    15,
    "right_ankle":   16,
}

KP_CONFIDENCE_THRESHOLD = 0.3  # below this a joint is considered invisible


@dataclass
class Keypoint:
    """A single joint in normalised frame coordinates (0-1)."""
    x: float        # normalised x (0 = left edge, 1 = right edge)
    y: float        # normalised y (0 = top edge, 1 = bottom edge)
    confidence: float

    @property
    def visible(self) -> bool:
        return self.confidence >= KP_CONFIDENCE_THRESHOLD


@dataclass
class Pose:
    """Bounding box + named keypoints for one detected person."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    keypoints: dict[str, Keypoint] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def kp(self, name: str) -> Keypoint | None:
        """Return a keypoint by name, or None if not available / invisible."""
        kp = self.keypoints.get(name)
        return kp if (kp and kp.visible) else None


@dataclass
class DetectionResult:
    poses: list[Pose] = field(default_factory=list)
    annotated_frame: np.ndarray | None = None

    @property
    def person_count(self) -> int:
        return len(self.poses)


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n-pose.pt", confidence: float = 0.4):
        log.info("Loading YOLOv8-Pose model: %s", model_name)
        self._model = YOLO(model_name)
        self._confidence = confidence
        log.info("Pose model ready (confidence threshold %.2f)", confidence)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run pose inference on a frame. Returns poses and annotated copy."""
        h, w = frame.shape[:2]
        results = self._model(frame, conf=self._confidence, verbose=False)[0]

        poses: list[Pose] = []

        if results.boxes is not None and results.keypoints is not None:
            kp_xyn = results.keypoints.xyn.cpu().numpy()   # (N, 17, 2) normalised
            kp_conf = results.keypoints.conf.cpu().numpy() # (N, 17)

            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                keypoints: dict[str, Keypoint] = {}
                for name, idx in KEYPOINT_NAMES.items():
                    kx, ky = float(kp_xyn[i, idx, 0]), float(kp_xyn[i, idx, 1])
                    kc = float(kp_conf[i, idx])
                    keypoints[name] = Keypoint(x=kx, y=ky, confidence=kc)

                poses.append(Pose(x1, y1, x2, y2, conf, keypoints))

                log.debug(
                    "Person %d | box=(%d,%d,%d,%d) | "
                    "l_hip=(%.2f,%.2f) r_hip=(%.2f,%.2f) | "
                    "l_knee=(%.2f,%.2f) r_knee=(%.2f,%.2f) | "
                    "l_wrist=(%.2f,%.2f) r_wrist=(%.2f,%.2f)",
                    i,
                    x1, y1, x2, y2,
                    keypoints["left_hip"].x,  keypoints["left_hip"].y,
                    keypoints["right_hip"].x, keypoints["right_hip"].y,
                    keypoints["left_knee"].x, keypoints["left_knee"].y,
                    keypoints["right_knee"].x,keypoints["right_knee"].y,
                    keypoints["left_wrist"].x,keypoints["left_wrist"].y,
                    keypoints["right_wrist"].x,keypoints["right_wrist"].y,
                )

        # results.plot() draws the skeleton overlay automatically.
        annotated = results.plot()

        return DetectionResult(poses=poses, annotated_frame=annotated)

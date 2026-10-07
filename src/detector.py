"""Modular motion detection and human pose estimation pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from src.config import MonitorConfig, ZoneDefinition
except ImportError:
    from config import MonitorConfig, ZoneDefinition


@dataclass
class ZoneStatus:
    """State of an individual monitored zone in a single frame."""
    name: str
    pts: np.ndarray
    movement_detected: bool = False
    human_detected: bool = False
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def is_alert(self) -> bool:
        """Returns True if any movement or human activity is detected."""
        return self.movement_detected or self.human_detected


class PoseTracker:
    """Estimates human body landmarks and evaluates intrusion into zones."""

    def __init__(self, min_detection_confidence: float = 0.8, min_tracking_confidence: float = 0.8):
        self.available = MEDIAPIPE_AVAILABLE
        self.pose = None

        if self.available:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            print("[WARN] MediaPipe is not installed. Pose estimation disabled.")

    def process_frame(self, frame_bgr: np.ndarray) -> List[Tuple[float, float]]:
        """Process a frame and return hip center coordinates for detected subjects.

        Args:
            frame_bgr: Input BGR frame.

        Returns:
            List of (x, y) coordinates representing hip centers.
        """
        if not self.available or self.pose is None:
            return []

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        coordinates = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]

            # Calculate mid-hip point in pixel coordinates
            hip_x = ((r_hip.x + l_hip.x) / 2.0) * w
            hip_y = ((r_hip.y + l_hip.y) / 2.0) * h
            coordinates.append((hip_x, hip_y))

        return coordinates

    def release(self) -> None:
        """Clean up MediaPipe resources."""
        if self.pose is not None:
            self.pose.close()


class ZoneMotionDetector:
    """Detects movement and human occupancy across configured polygonal zones."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.morph_kernel_size)

        # Initialize background subtractor
        if config.bg_subtractor_type.upper() == "MOG2":
            self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        else:
            try:
                self.fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
            except AttributeError:
                # Fallback to MOG2 if cv2.bgsegm is unavailable
                self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        # Optional pose estimator
        self.pose_tracker = PoseTracker(
            config.pose_confidence,
            config.pose_tracking_confidence
        ) if config.enable_pose else None

    def analyze_frame(self, frame: np.ndarray) -> List[ZoneStatus]:
        """Analyze a frame and return activity status for each defined zone.

        Args:
            frame: Resized BGR image frame.

        Returns:
            List of ZoneStatus instances for each configured zone.
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # Extract human body coordinates if pose tracking is active
        human_points = self.pose_tracker.process_frame(frame) if self.pose_tracker else []

        zone_statuses = []

        for zone in self.config.zones:
            pts = zone.get_pixel_points(width, height)
            status = ZoneStatus(name=zone.name, pts=pts)

            # Mask frame to zone polygon
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [pts], -1, 255, -1)
            zone_roi = cv2.bitwise_and(blurred, blurred, mask=mask)

            # Background subtraction and noise reduction
            fgmask = self.fgbg.apply(zone_roi)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
            fgmask = cv2.dilate(fgmask, None, iterations=self.config.dilate_iterations)

            # Find motion contours
            contours = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for cnt in contours:
                if cv2.contourArea(cnt) > zone.min_contour_area:
                    status.movement_detected = True
                    x, y, w, h = cv2.boundingRect(cnt)
                    status.bounding_boxes.append((x, y, w, h))

            # Test human presence using point-in-polygon
            for hx, hy in human_points:
                if cv2.pointPolygonTest(pts, (hx, hy), False) >= 0:
                    status.human_detected = True

            zone_statuses.append(status)

        return zone_statuses

    def draw_overlays(
        self,
        frame: np.ndarray,
        zone_statuses: List[ZoneStatus],
        fps_value: float = 0.0
    ) -> np.ndarray:
        """Annotate frame with zones, bounding boxes, labels, and telemetry."""
        output = frame.copy()
        h, w = output.shape[:2]

        # Draw HUD bar at top
        cv2.rectangle(output, (0, 0), (w, 45), (20, 20, 20), -1)

        hud_x = 10
        for i, status in enumerate(zone_statuses):
            zone_cfg = self.config.zones[i]
            color = zone_cfg.color_alert if status.is_alert else zone_cfg.color_normal

            # Draw zone polygon
            cv2.drawContours(output, [status.pts], -1, color, 2)

            # Draw bounding boxes for motion
            for x, y, bw, bh in status.bounding_boxes:
                cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 1)

            # Build status text
            state_desc = "EMPTY"
            if status.human_detected:
                state_desc = "HUMAN INTRUSION"
            elif status.movement_detected:
                state_desc = "MOTION"

            text = f"{status.name}: {state_desc}"
            cv2.putText(output, text, (hud_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            hud_x += int(w * 0.40)

        # Render FPS indicator in top-right
        fps_text = f"FPS: {int(fps_value)}"
        cv2.putText(output, fps_text, (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return output

    def release(self) -> None:
        """Release allocated detectors."""
        if self.pose_tracker:
            self.pose_tracker.release()

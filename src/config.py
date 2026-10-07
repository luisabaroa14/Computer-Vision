"""Configuration management for computer vision zone monitoring."""

import os
from dataclasses import dataclass, field
from typing import List, Tuple
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env file if available
load_dotenv()


@dataclass
class ZoneDefinition:
    """Represents a monitored polygonal region of interest (ROI)."""
    name: str
    points_relative: List[Tuple[float, float]]  # Points specified as relative fractions (x, y) in [0.0, 1.0]
    color_normal: Tuple[int, int, int] = (0, 200, 150)  # BGR
    color_alert: Tuple[int, int, int] = (0, 0, 255)     # BGR
    min_contour_area: int = 3000

    def get_pixel_points(self, width: int, height: int) -> np.ndarray:
        """Convert relative coordinates into pixel coordinates array for OpenCV."""
        pts = [[int(x * width), int(y * height)] for x, y in self.points_relative]
        return np.array(pts, dtype=np.int32)


@dataclass
class MonitorConfig:
    """Master application configuration."""
    # Video input source
    video_source: str = field(default_factory=lambda: os.getenv("VIDEO_SOURCE", "0"))

    # Processing resolution
    frame_width: int = int(os.getenv("FRAME_WIDTH", "854"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "480"))

    # Motion detection settings
    bg_subtractor_type: str = os.getenv("BG_SUBTRACTOR", "GMG")  # "GMG" or "MOG2"
    morph_kernel_size: Tuple[int, int] = (3, 3)
    dilate_iterations: int = 2

    # Human pose estimation settings
    enable_pose: bool = os.getenv("ENABLE_POSE", "true").lower() in ("true", "1", "yes")
    pose_confidence: float = 0.8
    pose_tracking_confidence: float = 0.8

    # Recording settings
    record_output: bool = os.getenv("RECORD_OUTPUT", "false").lower() in ("true", "1", "yes")
    output_video_path: str = os.getenv("OUTPUT_VIDEO_PATH", "output_videos/result.avi")
    output_fps: float = 15.0

    # Default configured zones (Key and Secondary areas from the original industrial setup)
    zones: List[ZoneDefinition] = field(default_factory=lambda: [
        ZoneDefinition(
            name="Key Area",
            points_relative=[
                (0.80, 1.00),
                (0.25, 1.00),
                (0.25, 0.70),
                (0.80, 0.70),
            ],
            color_normal=(150, 200, 0),
            color_alert=(0, 0, 255),
            min_contour_area=3000,
        ),
        ZoneDefinition(
            name="Secondary Area",
            points_relative=[
                (0.45, 0.525),
                (0.00, 0.675),
                (0.00, 0.250),
                (0.45, 0.250),
            ],
            color_normal=(150, 200, 0),
            color_alert=(0, 140, 255),
            min_contour_area=2500,
        ),
    ])

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return (width, height) resolution tuple."""
        return (self.frame_width, self.frame_height)

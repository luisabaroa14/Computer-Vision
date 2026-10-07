"""Threaded video stream reader for low-latency RTSP and camera capture."""

from threading import Lock, Thread
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class VideoStream:
    """Asynchronous video stream reader that fetches frames in a separate thread

    to eliminate OpenCV buffer lag for real-time processing.
    """

    def __init__(self, src: Union[int, str] = 0, name: str = "VideoStream"):
        """Initialize the video stream capture.

        Args:
            src: Camera device index (int), video file path (str), or RTSP URL (str).
            name: Thread identifier name.
        """
        self.src = src
        self.name = name
        self.stream = cv2.VideoCapture(src)

        if not self.stream.isOpened():
            print(f"[WARN] Failed to open video source: {src}")

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self._lock = Lock()
        self._thread: Optional[Thread] = None

    def start(self) -> "VideoStream":
        """Start the background thread to continuously fetch frames."""
        if self._thread is not None and self._thread.is_alive():
            return self

        self.stopped = False
        self._thread = Thread(target=self.update, name=self.name, daemon=True)
        self._thread.start()
        return self

    def update(self) -> None:
        """Continually read frames until stopped or stream ends."""
        while not self.stopped:
            if not self.stream.isOpened():
                break

            grabbed, frame = self.stream.read()
            with self._lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
                else:
                    self.stopped = True
                    break

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recently grabbed frame in a thread-safe manner."""
        with self._lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self.stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def release(self) -> None:
        """Stop thread and release OpenCV video capture resources."""
        self.stop()
        if self.stream.isOpened():
            self.stream.release()

    def __enter__(self) -> "VideoStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

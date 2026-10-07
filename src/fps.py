"""Frames-per-second (FPS) measurement utility."""

import time
from typing import Optional


class FPS:
    """Measures overall and rolling frames-per-second for video analysis pipelines."""

    def __init__(self, rolling_interval: float = 1.0):
        """Initialize FPS counter.

        Args:
            rolling_interval: Time window in seconds over which rolling FPS is computed.
        """
        self.rolling_interval = rolling_interval
        self._start: Optional[float] = None
        self._end: Optional[float] = None
        self._num_frames: int = 0

        # Rolling window attributes
        self._period_start: float = 0.0
        self._period_frames: int = 0
        self._current_fps: float = 0.0

    def start(self) -> "FPS":
        """Start the FPS timer."""
        self._start = time.time()
        self._period_start = self._start
        self._period_frames = 0
        self._num_frames = 0
        self._current_fps = 0.0
        self._end = None
        return self

    def stop(self) -> "FPS":
        """Stop the FPS timer."""
        self._end = time.time()
        return self

    def update(self) -> None:
        """Increment frame counters and recalculate rolling FPS if interval reached."""
        self._num_frames += 1
        self._period_frames += 1

        now = time.time()
        duration = now - self._period_start
        if duration >= self.rolling_interval:
            self._current_fps = self._period_frames / duration
            self._period_frames = 0
            self._period_start = now

    def elapsed(self) -> float:
        """Return elapsed time in seconds.

        If stop() has not been called, returns time elapsed since start().
        """
        if self._start is None:
            return 0.0
        end_time = self._end if self._end is not None else time.time()
        return max(end_time - self._start, 0.0)

    def mean_fps(self) -> float:
        """Return the mean frames-per-second over the total elapsed time."""
        total_time = self.elapsed()
        if total_time <= 0:
            return 0.0
        return self._num_frames / total_time

    @property
    def fps(self) -> float:
        """Get the current rolling frames-per-second."""
        return self._current_fps

    # Backward compatibility aliases
    def getMeanFps(self) -> float:
        """Backward-compatible alias for mean_fps."""
        return self.mean_fps()

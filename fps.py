import datetime
import time


class FPS:
    def __init__(self):
        # Store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

        # Store the number of periods of a frame and a counter
        self._periodFrames = 0
        self._counter = 0

    def start(self):
        # Start the timer
        self._start = datetime.datetime.now()
        self._period = time.time()
        return self

    def stop(self):
        # Stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # Increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # Return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def getMeanFps(self):
        # Return the mean frames per second
        return self._numFrames / self.elapsed()

    def fps(self):
        # Get the fps in the last period of time
        self._counter += 1
        if (time.time() - self._period) > 1:
            self._periodFrames = self._counter / (time.time() - self._period)
            self._counter = 0
            self._period = time.time()
        return self._periodFrames

# import the necessary packages
from threading import Thread
import cv2


class VideoStream:
    def __init__(self, src=0, name="VideoStream"):
        # Initialize the video stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # Initialize the thread name
        self.name = name

        # Initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # Loop until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # Otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the frame most recently read
        return self.grabbed, self.frame

    def release(self):
        # Indicate that the thread should be stopped
        self.stopped = True

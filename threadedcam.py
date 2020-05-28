from threading import Thread
import cv2, time

# https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def get_frame(self):
        cv2.imshow('frame', self.frame)
        return cv2.waitKey(self.FPS_MS)

if __name__ == '__main__':
    src = 'http://wzmedia.dot.ca.gov:1935/D3/80_donner_lake.stream/index.m3u8'
    threaded_camera = ThreadedCamera(src)
    while True:
        try:
            key = threaded_camera.get_frame()
        except AttributeError:
            pass
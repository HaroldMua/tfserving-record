import cv2
import queue
import threading
from base_camera import BaseCamera
from tfserving_inference import Detection

# Refer to: https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
# bufferless VideoCapture
class VideoCapture:

  def __init__(self, video_source):
    self.cap = cv2.VideoCapture(video_source)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader, daemon=True)
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
      if not self.cap.isOpened():
          raise RuntimeError('Could not start camera.')

      while True:
          ret, frame = self.cap.read()
          if not ret:
              break
          if not self.q.empty():
            try:
              self.q.get_nowait()   # discard previous (unprocessed) frame
            except queue.Empty:
              pass
          self.q.put(frame)

  def read(self):
    return self.q.get()


class Camera(BaseCamera):
    def __init__(self, feed_type, device, video_source_dict):
        super(Camera, self).__init__(feed_type, device, video_source_dict)

    @staticmethod
    def opencv_frames(device, video_source):
        # when 'video_source' is a number, it must be a int type.
        if len(video_source) < 3:
            video_source = int(video_source)

        # camera = cv2.VideoCapture(video_source)
        camera = VideoCapture(video_source)

        while True:
            frame = camera.read()
            # img = cv2.flip(img, 1)
            Detection.detect_object(frame)

            cam_id = device
            yield cam_id, frame

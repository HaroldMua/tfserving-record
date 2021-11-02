import imagezmq
from base_camera import BaseCamera
from tfserving_inference import Detection


class Camera(BaseCamera):
    def __init__(self, feed_type, device, video_source_dict):
        Camera.set_video_source_dict(video_source_dict)
        super(Camera, self).__init__(feed_type, device, video_source_dict)

    @staticmethod
    def set_video_source_dict(video_source_dict):
        Camera.video_source_dict = video_source_dict

    @staticmethod
    def ip_frames(device):

        image_hub = imagezmq.ImageHub(open_port='tcp://*:{}'.format(Camera.video_source_dict[device]))
        while True:
            _, frame = image_hub.recv_image()
            image_hub.send_reply(b'OK')  # this is needed for the stream to work with REQ/REP pattern
            Detection.detect_object(frame)

            cam_id = device
            yield cam_id, frame

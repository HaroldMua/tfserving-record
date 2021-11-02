import configparser
import numpy as np
import pickle
import requests
import sys
import time
from inference_client.object_detection.utils import visualization_utils as vis_util


class Detection:
    _CONFIG_FILE = "../config.ini"

    config = configparser.ConfigParser()
    config.read(_CONFIG_FILE)

    _TF_SERVING_URL = config["Tensorflow"]["tf_serving_url"]
    _FILE_LABELS = "coco"
    _THRESHOLD = 0.5

    @staticmethod
    def load_obj(name):
        with open('object_detection/data/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def expand_image(img):
        return np.expand_dims(img, axis=0)

    @staticmethod
    def preprocess_frame(frame):
        preprocessed_img = Detection.expand_image(frame)
        payload = {"instances": preprocessed_img.tolist()}
        return frame, preprocessed_img, payload

    @classmethod
    def detect_object(cls, frame):
        # load labels
        classes = cls.load_obj(Detection._FILE_LABELS)

        print("Starting detection")
        frame, preprocessed_img, payload = cls.preprocess_frame(frame)

        t0 = time.time()
        try:
            res = requests.post(
                Detection._TF_SERVING_URL,
                json=payload
            )
        except requests.exceptions.RequestException:
            print("ERROR: Request error, did you start Tensorflow Serving?")
            sys.exit()
        except Exception as e:
            raise e
        print("Amount of seconds to predict:", time.time() - t0)

        if (res.status_code == 400):
            print("Error:", res.text)
            pass
        else:
            t0 = time.time()
            output_dict = res.json()["predictions"][0]
            print("Amount of seconds to get JSON:", time.time() - t0)

            t0 = time.time()
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.array(output_dict['detection_boxes']),
                np.array(output_dict['detection_classes'], dtype="uint8"),
                output_dict['detection_scores'],
                classes,
                use_normalized_coordinates=True,
                line_thickness=2
            )
            print("Amount of seconds to visualize:", time.time() - t0)

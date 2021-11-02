import os
import configparser
import cv2
import numpy as np
import pickle
import queue
import requests
import sys
import threading
import time
import datetime
from pprint import pprint
from inference_client.object_detection.utils import visualization_utils as vis_util, output_utils as out_util


# -------------------
# FUNCTIONS
# -------------------

def save_obj(obj, name):
    with open(
        'object_detection/data/' + name + '.pkl',
        'wb'
    ) as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('object_detection/data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def expand_image(img):
    return np.expand_dims(img, axis=0)


def preprocess_frame(frame):
    preprocessed_img = expand_image(frame)
    payload = {"instances": preprocessed_img.tolist()}
    return frame, preprocessed_img, payload


def save_detection(classname, image):
    """Saves a image
    :classname: name of the class
    :image: image to save
    """
    date = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
    os.makedirs(f"{_SAVE_DIR}/{classname}", exist_ok=True)

    return cv2.imwrite(f"{_SAVE_DIR}/{classname}/{date}-{classname}.jpg", image)

# ------------------
# CONFIGURATION
# ------------------

_CONFIG_FILE = "../config.ini"

config = configparser.ConfigParser()
config.read(_CONFIG_FILE)

_TF_SERVING_URL = config["Tensorflow"]["tf_serving_url"]
_FILE_LABELS = "coco"
_THRESHOLD = 0.5

_SAVE_DETECTION = config["General"].getboolean("saved_img")
_SAVE_DIR = config["General"]["save_dir"]

_SKIP_FRAMES = config['General'].getboolean("skip_frames")


# ------------------
# VIDEOSTREAM FEED URL
# ------------------

if config['IPCapture']['IP'] != 'no':
    _DETECTION_SOURCE = config['IPCapture']['IP'] # ip摄像头
elif config['USBCapture']['USB'] != 'no':
    _DETECTION_SOURCE = 0  # USB摄像头或采集卡设备
elif config['PiCamera']['PI'] != 'no':
    _DETECTION_SOURCE = 1 # 树莓派
elif  config['VideoPath']['PATH'] != 'no':
    _DETECTION_SOURCE = config['VideoPath']['PATH'] # 本地视频源


# -------------------
# INITIALISATION
# -------------------

cap = cv2.VideoCapture(
    _DETECTION_SOURCE
)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ret, frame = cap.read(1)
_HEIGHT, _WIDTH, _ = frame.shape

if not cap.isOpened():
    print("Cannot read video stream, exiting...")
    sys.exit(1)

# load labels
classes = load_obj(_FILE_LABELS)


# ------------------
# QUEUE & THREADING RETRIEVING
# ------------------

"""
python的queue模块提供了：FIFO（先入先出)队列Queue，LIFO（后入先出）队列LifoQueue，和优先级队列PriorityQueue
"""

# 先进后出（LIFO）队列, 类似于栈
frame_queue = queue.LifoQueue(5)

lock = threading.Lock()
frame_counter = 0

# 获取图像帧
def retrieve_frames(cap):
    global frame_counter
    # frame_counter = 0

    print("Retrieving frames")
    while retrieving_frames:
        t0 = time.time()
        ret, frame = cap.read(1)
        print("Amount of seconds to get frame:", time.time() - t0)
        print(f"[id: {frame_counter}] Got frame")

        if frame is None:
            time.sleep(0.2)
        else:
            frame_queue.put(preprocess_frame(frame))
            print(f"[id: {frame_counter}] Preprocessed frame")

        while frame_queue.full() and _SKIP_FRAMES:
            cap.grab()    # 函数cv2.VideoCapture.grab()用来指向下一帧
            # 在with语句中使用锁Lock，就不用显性地使用acquire()和release()方法
            with lock:
                frame_counter += 0
            # lock.acquire()
            # frame_counter += 0
            # lock.release()

        lock.acquire()   # 加锁，保证线程安全
        frame_counter += 1   # 修改frame_counter对象
        lock.release()   # 释放锁


# 使用一个守护进程，执行获取图像帧任务
for i in range(1):   # simple and funny loop, haha
    retrieving_frames = True
    th_retrieve_frames = threading.Thread(
        target=retrieve_frames,
        # name=f'thread-retrieve-{i}',
        kwargs={"cap": cap},
        daemon=True
    )

    th_retrieve_frames.start()
    time.sleep(1)


# ------------------
# QUEUE & THREADING DETECTIONS
# ------------------

detection_queue = queue.Queue()

def handle_detections():
    """
    if necessary, save the detected frames according to specific classes
    """

    vehicles = [
        "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "skateboard"
    ]

    animals = [
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe"
    ]

    while True:
        frame, detections = detection_queue.get()
        if len(detections) > 0:
            print(f"Got detections at {time.ctime()}:")
            pprint(detections)

            # save specific classes images
            for detection in detections:
                if detection["class"] in vehicles:
                    if _SAVE_DETECTION:
                        save_detection(detection["class"], frame)
                elif detection["class"] in animals:
                    if _SAVE_DETECTION:
                        save_detection(detection["class"], frame)
                elif detection["class"] in "person":
                    if _SAVE_DETECTION:
                        save_detection(detection["class"], frame)

        """
        Queue.task_done():
        在完成一项工作之后，Queue.task_done()函数向任务已经完成的队列发送一个信号
        """

        detection_queue.task_done()


# 使用一个守护进程，执行保存图像帧任务
th_detections = threading.Thread(
    target=handle_detections,
    # name=f"thread-detect-{i}",
    daemon=True
)

th_detections.start()


# -------------------
# PROCESSING
# -------------------

print("Starting detection")
while True:
    # frame video file
    print("New frame queue item. Amount of frames in queue:",
          frame_queue.qsize())

    frame, preprocessed_img, payload = frame_queue.get()
    frame_queue.task_done()

    t0 = time.time()
    try:
        res = requests.post(
            _TF_SERVING_URL,
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
            # instance_masks=output_dict['detection_masks'],
            use_normalized_coordinates=True,
            line_thickness=2
        )
        print("Amount of seconds to visualize:", time.time() - t0)


    # show frame to user
    t0 = time.time()
    # cv2.imshow('frame', frame)
    cv2.imshow('frame', preprocessed_img[0])
    print("Amount of seconds to show image:", time.time() - t0)


    detections = out_util.convert_output_to_detections(
        output_dict, classes, _THRESHOLD, _WIDTH, _HEIGHT)
    detection_queue.put((frame, detections))
    # detection_queue.put((preprocessed_img[0], detections))


    # close windows when pressing 'q'
    if cv2.waitKey(50) & 0xFF == ord('q'):
        retrieving_frames = False
        cv2.destroyAllWindows()

        # exit script
        print("Exiting script")
        sys.exit(0)


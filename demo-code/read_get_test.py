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


frame_queue = queue.LifoQueue(5)
detection_queue = queue.Queue()

lock = threading.Lock()

def read_frames(cap):
    frame_counter = 0

    print("Getting frames")
    while True:
        ret, frame = cap.read(1)
        print(f"[id: {frame_counter}] Got frame.")

        if frame is None:
            time.sleep(0.2)
        else:
            frame_queue.put(frame)
            print(f"[id: {frame_counter}] Put frame to frame_queue.")

        while frame_queue.full():
            cap.grab()
            with lock:
                frame_counter += 0

        with lock:
            frame_counter += 1



def get_frames():
    pass



if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("test.mp4")

    th_read_frames = threading.Thread(
        target=read_frames,
        kwargs={"cap": cap},
        daemon=True
    )
    th_read_frames.start()
    time.sleep(1)

    # close windows when pressing 'q'
    if cv2.waitKey(50) & 0xFF == ord('q'):
        retrieving_frames = False
        cv2.destroyAllWindows()

        # exit script
        print("Exiting script")
        sys.exit(0)





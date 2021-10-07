from __future__ import print_function
from utils.app_utils import *
import multiprocessing
from multiprocessing import Queue, Pool
import cv2

def realtime(args):
    """
    Read and apply object detection to input real time stream (webcam)
    """

    


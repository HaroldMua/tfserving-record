import configparser
import cv2
from imutils.video import VideoStream # 该包依赖于NumPy、Opencv和matplotlib, 在opencv基础上对一些方法进行了再次加工


config = configparser.ConfigParser()
config.read('config.ini')
if config['IPCapture']['IP'] != 'no':
    # vs = VideoStream(src= config['IPCapture']['IP']).start()  # ip摄像头
    vs = cv2.VideoCapture(config['IPCapture']['IP']) # ip摄像头
elif config['USBCapture']['USB'] != 'no':
    # vs = VideoStream(src=0).start()  # USB摄像头或采集卡设备
    vs = cv2.VideoCapture(0)  # USB摄像头或采集卡设备
elif config['PiCamera']['PI'] != 'no':
    # vs = VideoStream(usePiCamera=1).start()  # 树莓派
    vs = cv2.VideoCapture(1) # 树莓派
elif  config['VideoPath']['PATH'] != 'no':
    # vs = VideoStream(src="test.mp4").start()  # 本地视频源
    vs = cv2.VideoCapture("test.mp4") # 本地视频源



while True:
    # frame = vs.read()
    ret, frame = vs.read()
    frame = cv2.flip(frame,1)   #镜像操作
    cv2.imshow("video", frame)
    key = cv2.waitKey(50)
    # print(key)
    if key  == ord('q'):  #判断是哪一个键按下
        break
cv2.destroyAllWindows()


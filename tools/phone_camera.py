# Refer to: https://blog.csdn.net/weixin_41010198/article/details/106100497

import cv2
from imutils.video import VideoStream
import imagezmq


# url = "http://admin:admin@192.168.11.103:8081"
# url = "http://admin:123456@192.168.3.16:8081"

url = "rtsp://admin:admin@192.168.11.103:8554/live"
# url = "rtsp://admin:123456@192.168.3.16:8554/live"


print('start')
cap = cv2.VideoCapture(url)#读取视频流
# cap = cv2.VideoCapture('inference_client/data_source/video/test.mp4')
# while(cap.isOpened()):

while(cap.isOpened()):

    ret, frame = cap.read()
    print('success')
    cv2.imshow('frame', frame)
    cv2.waitKey(10)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


cap.release()
cv2.destroyAllWindows()

# cap = VideoStream(0).start()  #读取视频流
#
# sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')  # change to IP address and port of server thread
# cam_id = 'Camera 1'  # this name will be displayed on the corresponding camera stream
#
# stream = cap.start()
#
# while True:
#
#     frame = stream.read()
#     sender.send_image(cam_id, frame)


# print('start')
# cap = VideoStream(0).start()  #读取视频流
# while True:
#     frame = cap.read()
#     print('success')
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

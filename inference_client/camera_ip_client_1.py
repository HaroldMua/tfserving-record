from imutils.video import VideoStream
import imagezmq
import cv2
import time

cap = VideoStream(0).start()

sender = imagezmq.ImageSender(connect_to='tcp://192.168.11.216:5555')  # change to IP address and port of server thread
cam_id = 'Camera 1'  # this name will be displayed on the corresponding camera stream
time.sleep(2)

id = 0
while True:
    frame = cap.read()
    sender.send_image(cam_id, frame)

    id += 1
    print("frame id: %d" % id)

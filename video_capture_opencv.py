import numpy as np
import cv2

def expand_image(img):
    return np.expand_dims(img, axis=0)

capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame,1)   #镜像操作

    preprocessed_img = expand_image(frame)

    cv2.imshow("video", preprocessed_img[0])

    cv2.imshow("frame", frame)

    key = cv2.waitKey(50)
    print(key)
    if key  == ord('q'):  #判断是哪一个键按下
        break
cv2.destroyAllWindows()

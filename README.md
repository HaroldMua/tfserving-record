# Multi-Camera Live Object Detection with Tensorflow Serving

## 如何使用异步进程获取IP摄像头的视频流

[ImageZMQ](https://github.com/jeffbass/imagezmq) 可以实现，多个sender(Raspberry Pi)的视频流发送至
reciever(PC)。

例如，可在多个Raspberry Pi运行以下代码，发送视频流：
```
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
```

在PC端运行以下代码，接收视频流：
```
import cv2
import imagezmq

image_hub = imagezmq.ImageHub()

while True:  # show streamed images until Ctrl-C
    rpi_name, image = image_hub.recv_image()
    cv2.imshow(rpi_name, image)  # 1 window for each RPi

    cv2.waitKey(1)
    image_hub.send_reply(b'OK')

```

## 代码特点
- 通过定义[config.ini](config.ini),读取Tensorflow Serving的URL配置信息
- flask视频流的实现参考[flask-video-streaming](https://github.com/miguelgrinberg/flask-video-streaming),
  该部分代码是python类之静态方法、类方法的优秀实现
- 




Record the walkthrough use of tfserving.

- [Tensorflow Serving](docs/tensorflow_serving_with_docker.md) shows you how to use TensorFlow Serving with Docker(CPU/GPU).
- [serving_basic](serving_basic.md) shows you how to use TensorFlow Serving components to export a trained TensorFlow model(as SavedModel format) and use a Docker serving image to easily load the model for serving.
- [convert_model_to_TFserving](docs/convert_model_to_TFserving.md)

## Deploy a Tensorflow model wiht TF Serving

- [Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Deployment of TensorFlow models into production with TensorFlow Serving, Docker, Kubernetes and Microsoft Azure](https://github.com/Vetal1977/tf_serving_example)
- [How to deploy an object detection model with tensorflow serving](https://www.freecodecamp.org/news/how-to-deploy-an-object-detection-model-with-tensorflow-serving-d6436e65d1d9/)
- [Bravo!!! Deploy tensorflow object detection model in less than 5 minutes](https://pierrepaci.medium.com/deploy-tensorflow-object-detection-model-in-less-than-5-minutes-604e6bb0bb04)


## Deploy a Pytorch model wiht TF Serving

- [yolov5 model downloads](https://github.com/ultralytics/yolov5/releases)


## 使用TensorFlow Serving镜像的两种方式

- 使用tensorflow/serving基础镜像，将模型文件挂载在对应文件夹
- 使用基于tensorflow/serving基础镜像的自定义镜像，将模型文件放置在镜像中


## Reference

- [微信公众号-NLP奇幻之旅，系列文章：利用TFserving实现模型部署及预测](https://mp.weixin.qq.com/s?src=11&timestamp=1633408156&ver=3355&signature=nypzJ7FC6vY7mSkG46ctafMVb5hj3TuaTmcNdbcp1UbtG9NywFNZGgHtt1G6bNbBK6N24Viy26Vkidi6VnWlr7uT9wM2*Ec9qEQs7U1NBM2S8TrUlrrzq2j-leWv7FXY&new=1)
- [微信公众号-Python中文社区，利用TFserving部署深度学习模型](https://mp.weixin.qq.com/s?src=11&timestamp=1633408370&ver=3355&signature=rqM2BY3HnMQz3pJ7wiDUK-M3hqr4Yudx-c*JhHPHIaLDdD9GUPBHSrD9RxWwO3axwKEULcJEKvJo1gXNe4gsI3JGBjMy2fiq-RmW5-kqiunBD6Joy*Y3crBOj2tSLPQE&new=1)
- [通过keras将ptorch .pth模型转tensorflow .pb模型](https://blog.csdn.net/pinggengxiu5246/article/details/104041386)
- [ptorch2keras](https://github.com/gmalivenko/pytorch2keras)
- [pytorch-onnx-tensorflow-pb](https://github.com/cinastanbean/pytorch-onnx-tensorflow-pb)
- [pytorch保存网络结构及参数](https://blog.csdn.net/qq_40520596/article/details/106955452)
- [tesorflow三种模型的加载和保存方法](https://blog.csdn.net/weixin_44388679/article/details/107458536)
- [pytorch加载本地和官方预训练模型](https://blog.csdn.net/weixin_36474809/article/details/89646008)
- [yolov5-pytorch -> onnx -> tf savemodel](https://blog.csdn.net/qq_36756866/article/details/116834551)
- [tensorflow-yolov3 serving](https://github.com/Byronnar/tensorflow-serving-yolov3)
- [pytorch2tf](https://github.com/yxlee245/pytorch2tf)
- [Deploying Object Detection Model with TensorFlow Serving](https://medium.com/innovation-machine/deploying-object-detection-model-with-tensorflow-serving-7f12ee59b036)
- [tf-chpt-2-pb](https://github.com/r1cebank/tf-ckpt-2-pb)


- [camera-feed-object-detector-tf-serve](https://github.com/LanderMoerkerke/camera-feed-object-detector-tf-serve)
- [darknet-tensorflow-serving](darknet-tensorflow-serving)
- [tensorflow_serving_examples](https://github.com/percent4/tensorflow_serving_examples)


- [html5播放rtsp方案](https://blog.csdn.net/u014535295/article/details/99303890)

## github projects:
- [opencv-flask](https://github.com/DataXujing/OpenCV-Flask)
  - 将目标检测程序生成为系统的动态链接库文件
  - 用flask调用opencv和yolo-lite
- [yolov5-flask-vue](https://github.com/Sharpiless/Yolov5-Flask-VUE)

- https://github.com/lbeaucourt/Object-detection
- [object-detection](https://github.com/cristianpb/object-detection)
  - [Blog](https://cristianpb.github.io/blog/ssd-yolo)
- https://github.com/LanderMoerkerke/camera-feed-object-detector-tf-serve

- [flask-video-streaming](https://github.com/miguelgrinberg/flask-video-streaming)
  - [Blog](https://blog.miguelgrinberg.com/post/flask-video-streaming-revisited)
  
- [flask-opencv-streaming](https://github.com/desertfury/flask-opencv-streaming)

- [opencv-stream-video-to-web](https://github.com/pornpasok/opencv-stream-video-to-web)

- [live-video-stream](https://github.com/sgino209/live_video_stream)

- [multi-camera-live-object-tracking](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking)





### Client




## Tips


## TODO

将coco label从pkl文件读写并保存



client端的代码逻辑是不错的：
- 提取视频帧有一个守护进程
- 保存检测的视频帧有一个守护进程
- 主程序执行预测处理

优化：
- tf serving需要warm up?
- 应选择挂载保存在本地的模型的方式？


收获：
- 在feed.py中，学习使用了锁，线程

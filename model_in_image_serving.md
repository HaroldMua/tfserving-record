# Model in images seving

This tutorial shows you how to build a custom image with wrapping DL model and use a client to test the API.

## Dockerfile
```Dockerfile
FROM tensorflow/serving

# Define metadata
LABEL author="Pierre Paci"
LABEL version="1.0"
LABEL description="Deploy tensorflow object detection model by url"

# install wget
RUN apt-get update
RUN apt-get install -qy wget

# Create variable. Use it with docker build --build-arg model_url=...
ARG model_url

# Download model
WORKDIR /models
RUN wget -nv -O model.tar.gz $model_url
RUN tar -xvf model.tar.gz
RUN mkdir -p object-detect/1
RUN find -name saved_model -exec mv {}/saved_model.pb {}/variables object-detect/1/ \;

# Expose ports
# gRPC
EXPOSE 8080

# REST
EXPOSE 8081

ENTRYPOINT ["tensorflow_model_server", "--model_base_path=/models/object-detect"]
CMD ["--rest_api_port=8080","--port=8081"]
```

## Build & Run custm image

In this example, we will use [ssd_resnet_50_fpn_coco](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz), 
it will provide us a very good baseline, which is both fast and accurate. 

```shell
# Replace 'object_detection' with the name you want
# Replace MODEL_URL by the choosen model url
# e.g. MODEL_URL = http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
docker build -t object_detection --build-arg model_url=MODLET_URL -f tensorflow-object-serving.Dockerfile .

docker run -t --rm  -p 8080:8080 -p 8081:8081 object-detect &
```

## Client

```python
import PIL.Image
import numpy
import requests
from pprint import pprint
import time

image = PIL.Image.open("dog.jpg")  # Change dog.jpg with your image
image_np = numpy.array(image)


payload = {"instances": [image_np.tolist()]}
start = time.perf_counter()
res = requests.post("http://localhost:8080/v1/models/default:predict", json=payload)
print(f"Took {time.perf_counter()-start:.2f}s")
pprint(res.json())
```

# Refer to https://pierrepaci.medium.com/deploy-tensorflow-object-detection-model-in-less-than-5-minutes-604e6bb0bb04
# Running example:
# Replace 'object_detection' with the name you want
# Replace MODEL_URL by the choosen model url
# e.g. MODEL_URL = http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
# docker build -t object_detection --build-arg model_url=MODLET_URL -f tensorflow-object-serving.Dockerfile .

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

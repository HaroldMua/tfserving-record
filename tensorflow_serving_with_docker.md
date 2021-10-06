- [TensorFlow Serving with Docker](#TensorFlow Serving with Docker)
- [Wrapping the model in image](#Wrapping the model in image)
- [Mounting the model](#Mounting the model)


# TensorFlow Serving with Docker

## Serving with Docker

<pre class="prettyprint lang-bsh">
# Download the TensorFlow Serving Docker image and repo
<code class="devsite-terminal">docker pull tensorflow/serving</code><br/>
<code class="devsite-terminal">git clone https://github.com/tensorflow/serving</code>
# Location of demo models
<code class="devsite-terminal">TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"</code>

# Start TensorFlow Serving container and open the REST API port
<code class="devsite-terminal">docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &</code>

# Query the model using the predict API
<code class="devsite-terminal">curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict</code><br/>
# Returns => { "predictions": [2.5, 3.0, 4.5] }
</pre>


## Serving with Docker using GPU

<pre class="prettyprint lang-bsh">
# Download the TensorFlow Serving Docker image and repo
<code class="devsite-terminal">docker pull tensorflow/serving:2.0.0-gpu</code><br/>
<code class="devsite-terminal">git clone https://github.com/tensorflow/serving</code>
# Location of demo models
<code class="devsite-terminal">TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"</code>

# Start TensorFlow Serving container and open the REST API port
<code class="devsite-terminal">docker run --gpus all -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_gpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving:2.0.0-gpu &</code>

# Query the model using the predict API
<code class="devsite-terminal">curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict</code><br/>
# Returns => { "predictions": [2.5, 3.0, 4.5] }
</pre>

# Wrappin the model in image

There are two ways to deploy the model:
- wrapping the model in image
- mounting the model when starting the container.

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

# Mounting the model

The test passed!!!

This tutorial shows you how to use TensorFlow Serving components to export a
trained TensorFlow model and use the standard tensorflow_model_server to serve
it. If you are already familiar with TensorFlow Serving, and you want to know
more about how the server internals work, see the
[TensorFlow Serving advanced tutorial](serving_advanced.md).

This tutorial uses a simple Softmax Regression model that classifies handwritten
digits. It is very similar to the one introduced in the
[TensorFlow tutorial on image classification using the Fashion MNIST dataset](https://www.tensorflow.org/tutorials/keras/classification).

The code for this tutorial consists of two parts:

*   A Python file,
    [mnist_saved_model.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_saved_model.py),
    that trains and exports the model.

*   A ModelServer binary which can be either installed using Apt, or compiled
    from a C++ file
    ([main.cc](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/model_servers/main.cc)).
    The TensorFlow Serving ModelServer discovers new exported models and runs a
    [gRPC](http://www.grpc.io) service for serving them.

Before getting started, first [install Docker](tensorflow_serving_with_docker.md#installing-docker).

## Train and export TensorFlow model

For the training phase, the TensorFlow graph is launched in TensorFlow session
`sess`, with the input tensor (image) as `x` and output tensor (Softmax score)
as `y`.

Then we use TensorFlow's [SavedModelBuilder module](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py)
to export the model. `SavedModelBuilder` saves a "snapshot" of the trained model
to reliable storage so that it can be loaded later for inference.

For details on the SavedModel format, please see the documentation at
[SavedModel README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).

From [mnist_saved_model.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_saved_model.py),
the following is a short code snippet to illustrate the general process of
saving a model to disk.

```python
export_path_base = sys.argv[-1]
export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(FLAGS.model_version)))
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(
    sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_images':
            prediction_signature,
        tf.compat.v1.saved_model.signature_constants
            .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            classification_signature,
    },
    main_op=tf.compat.v1.tables_initializer(),
    strip_default_attrs=True)
builder.save()
```

`SavedModelBuilder.__init__` takes the following argument:

* `export_path` is the path of the export directory.

`SavedModelBuilder` will create the directory if it does not exist. In the
example, we concatenate the command line argument and `FLAGS.model_version` to
obtain the export directory. `FLAGS.model_version` specifies the **version** of
the model. You should specify a larger integer value when exporting a newer
version of the same model. Each version will be exported to a different
sub-directory under the given path.

You can add meta graph and variables to the builder using
`SavedModelBuilder.add_meta_graph_and_variables()` with the following arguments:

*   `sess` is the TensorFlow session that holds the trained model you are
    exporting.

*   `tags` is the set of tags with which to save the meta graph. In this case,
    since we intend to use the graph in serving, we use the `serve` tag from
    predefined SavedModel tag constants. For more details, see
    [tag_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py)
    and
    [related TensorFlow API documentation](https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/tag_constants).

*   `signature_def_map` specifies the map of user-supplied key for a
    **signature** to a tensorflow::SignatureDef to add to the meta graph.
    Signature specifies what type of model is being exported, and the
    input/output tensors to bind to when running inference.

    The special signature key `serving_default` specifies the default serving
    signature. The default serving signature def key, along with other constants
    related to signatures, are defined as part of SavedModel signature
    constants. For more details, see
    [signature_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)
    and
    [related TensorFlow API documentation](https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/signature_constants).

    Further, to help build signature defs easily, the SavedModel API provides
    [signature def utils](https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/signature_def_utils)..
    Specifically, in the original
    [mnist_saved_model.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_saved_model.py)
    file, we use `signature_def_utils.build_signature_def()` to build
    `predict_signature` and `classification_signature`.

    As an example for how `predict_signature` is defined, the util takes the
    following arguments:

    *   `inputs={'images': tensor_info_x}` specifies the input tensor info.

    *   `outputs={'scores': tensor_info_y}` specifies the scores tensor info.

    *   `method_name` is the method used for the inference. For Prediction
        requests, it should be set to `tensorflow/serving/predict`. For other
        method names, see
        [signature_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)
        and
        [related TensorFlow API documentation](https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/signature_constants).

Note that `tensor_info_x` and `tensor_info_y` have the structure of
`tensorflow::TensorInfo` protocol buffer defined
[here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto).
To easily build tensor infos, the TensorFlow SavedModel API also provides
[utils.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/utils.py),
with
[related TensorFlow API documentation](https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/utils).

Also, note that `images` and `scores` are tensor alias names. They can be
whatever unique strings you want, and they will become the logical names
of tensor `x` and `y` that you refer to for tensor binding when sending
prediction requests later.

For instance, if `x` refers to the tensor with name 'long_tensor_name_foo' and
`y` refers to the tensor with name 'generated_tensor_name_bar', `builder` will
store tensor logical name to real name mapping ('images' ->
'long_tensor_name_foo') and ('scores' -> 'generated_tensor_name_bar').  This
allows the user to refer to these tensors with their logical names when
running inference.

Note: In addition to the description above, documentation related to signature
def structure and how to set up them up can be found [here](signature_defs.md).

Let's run it!

First, if you haven't done so yet, clone this repository to your local machine:

```shell
git clone https://github.com/tensorflow/serving.git
cd serving
```

Clear the export directory if it already exists:

```shell
rm -rf /tmp/mnist
```

Now let's train the model:

```shell
tools/run_in_docker.sh python tensorflow_serving/example/mnist_saved_model.py \
  /tmp/mnist
```

This should result in output that looks like:

```console
Training model...

...

Done training!
Exporting trained model to models/mnist
Done exporting!
```

Now let's take a look at the export directory.

```console
$ ls /tmp/mnist
1
```

As mentioned above, a sub-directory will be created for exporting each version
of the model. `FLAGS.model_version` has the default value of 1, therefore
the corresponding sub-directory `1` is created.

```console
$ ls /tmp/mnist/1
saved_model.pb variables
```

Each version sub-directory contains the following files:

  * `saved_model.pb` is the serialized tensorflow::SavedModel. It includes
  one or more graph definitions of the model, as well as metadata of the
  model such as signatures.

  * `variables` are files that hold the serialized variables of the graphs.

With that, your TensorFlow model is exported and ready to be loaded!

## Load exported model with standard TensorFlow ModelServer

Use a Docker serving image to easily load the model for serving:

```shell
docker run -p 8500:8500 \
--mount type=bind,source=/tmp/mnist,target=/models/mnist \
-e MODEL_NAME=mnist -t tensorflow/serving &
```

Attention!!! Image `tensorflow/serving:latest-gpu` can not be used for serving, because the inference model was trained using cpu version of Tensorflow.

## Test the server

We can use the provided
[mnist_client](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_client.py)
utility to test the server. The client downloads MNIST test data, sends them as
requests to the server, and calculates the inference error rate.


```shell
tools/run_in_docker.sh python tensorflow_serving/example/mnist_client.py \
  --num_tests=1000 --server=127.0.0.1:8500
```

This should output something like

```console
    ...
    Inference error rate: 11.13%
```

We expect around 90% accuracy for the trained Softmax model and we get 11%
inference error rate for the first 1000 test images. This confirms that the
server loads and runs the trained model successfully!

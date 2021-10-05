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

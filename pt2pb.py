import logging
import argparse

import torch
import sys

def main():
    # sys.path.insert(0, "./model")


    # model = torch.load('./model/yolov5s.pt', map_location=torch.device('cpu'))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--onnx-file', help="File where to export the ONNX file", type=str,
    #     required=True)
    # parser.add_argument(
    #     '--meta-file', help="File where to export the Tensorflow meta file",
    #     type=str, required=True)
    # parser.add_argument(
    #     '--export-dir',
    #     help="Folder where to export proto models for TF serving",
    #     type=str, required=True)

    # args = parser.parse_args()
    # main(args)

    main()

"""

# https://towardsdatascience.com/converting-a-simple-deep-learning-model-from-pytorch-to-tensorflow-b6b353351f5d

model_pytorch = SimpleModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
model_pytorch.load_state_dict(torch.load('./models/model_simple.pt'))

dummy_input = torch.from_numpy(X_test[0].reshape(1, -1)).float().to(device)
dummy_output = model_pytorch(dummy_input)
print(dummy_output)

# Export to ONNX format
torch.onnx.export(model_pytorch, dummy_input, './models/model_simple.onnx', input_names=['test_input'], output_names=['test_output'])


# Load ONNX model and convert to TensorFlow format
model_onnx = onnx.load('./models/model_simple.onnx')

tf_rep = prepare(model_onnx)

# Export model as .pb file
tf_rep.export_graph('./models/model_simple.pb')


tf_graph = load_pb('./models/model_simple.pb')
sess = tf.Session(graph=tf_graph)

# Show tensor names in graph
for op in tf_graph.get_operations():
  print(op.values())

output_tensor = tf_graph.get_tensor_by_name('test_output:0')
input_tensor = tf_graph.get_tensor_by_name('test_input:0')

output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
print(output)

"""

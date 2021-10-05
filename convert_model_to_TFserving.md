# Convert model to TFserving

TFserving的模型需要转换成TFserving的格式， 不支持通常的checkpoint和pb格式。TFserving的模型[SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)
包含一个.pb文件和variables目录（可以为空）,导出格式如下：
```
./saved_model
├── 1
│   ├── assets
│   ├── asserts.extra
│   ├── variables
│   │        ├── variables.data-?????-of-?????
│   │        └── variables.index
│   └── saved_model.pb
├── 2
│   ├── assets
│   ├── asserts.extra
│   ├── variables
│   │        ├── variables.data-?????-of-?????
│   │        └── variables.index
│   └── saved_model.pb
```
SavedModel 是一个包含序列化签名和运行这些签名所需的状态的目录，其中包括变量值和词汇表。saved_model.pb 文件用于存储实际 TensorFlow 程序或模型，以及一组已命名的签名——每个签名标识一个接受张量输入和产生张量输出的函数。
variables 目录包含一个标准训练检查点。assets 目录包含 TensorFlow 计算图使用的文件，例如，用于初始化词汇表的文本文件。SavedModel 可能有一个用于保存 TensorFlow 计算图未使用的任何文件的 assets.extra 目录，例如，为使用者提供的关于如何处理 SavedModel 的信息。TensorFlow 本身并不会使用此目录。



不同的深度学习框架的转换路径：
```
pytorch(.pth)--> onnx(.onnx)--> tensorflow(.pb) --> TFserving
keras(.h5)--> tensorflow(.pb) --> TFserving
tensorflow(.pb) --> TFserving(TF 1.X和TF 2.X用的API不同)
```

## Tensorflow to Tensorflow serving

## Pytorch to Tensorflow serving

## Reference

- [pytorch转onnx](https://pytorch.org/docs/1.9.1/onnx.html)
- [keras转TFserving](https://www.tensorflow.org/guide/saved_model)


"""
Refer to: https://mp.weixin.qq.com/s?src=11&timestamp=1633333359&ver=3353&signature=j1H7zt1AZcFuVSqCe0NxHjQ05cRhlVvSWZWYai5shLkL5DbCu5sZOVvDr2G89OmldMz*VXBPf2OKJ5bQqZyuGvIfPPfOGDb1JzPYPPlYdhHGUiExvDiWjYwvhr4suju7&new=1

"""

import tensorflow as tf

def create_graph(pb_file):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def pb_to_tfserving(pb_file, export_path, pb_io_name=[], input_node_name='input', output_node_name='output',
                    signature_name='default_tfserving'):
    # pb_io_name 为 pb模型输入和输出的节点名称，
    # input_node_name为转化后输入名
    # output_node_name为转化后输出名
    # signature_name 为签名
    create_graph(pb_file)
    # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    input_name = '%s:0' % pb_io_name[0]
    output_name = '%s:0' % pb_io_name[1]
    with tf.Session() as sess:
        in_tensor = sess.graph.get_tensor_by_name(input_name)
        out_tensor = sess.graph.get_tensor_by_name(output_name)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)  ## export_path导出路径
        inputs = {input_node_name: tf.saved_model.utils.build_tensor_info(in_tensor)}
        outputs = {output_node_name: tf.saved_model.utils.build_tensor_info(out_tensor)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs, outputs, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        builder.add_meta_graph_and_variables(
            sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={signature_name: signature}, clear_devices=True)  ## signature_name为签名，可自定义
        builder.save()


pb_model_path = 'test.pb'
pb_to_tfserving(pb_model_path, './1', pb_io_name=['input_1_1', 'output_1'], signature_name='your_model')


def export_for_serving()

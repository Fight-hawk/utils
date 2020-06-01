import tensorflow as tf

def freeze_graph_ckpt(model_folder, id, output_node_names, output_graph):

    input_checkpoint = model_folder + "-" + id

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:

        saver.restore(sess, input_checkpoint)

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))
    for op in graph.get_operations():
        print(op.name, op.values())

def freeze_graph_savedModel(model_folder, output_node_names, output_graph):

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], model_folder)

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))
    for op in graph.get_operations():
        print(op.name, op.values())

def main():
    #ckpt
    model_folder = 'D:\PyProject\MMNet\output\ESPNetModel'
    id = '158000'
    output_node_names = 'ESPNet/seg_net/feather/sigmoid'
    output_graph = 'model.pb'

    freeze_graph_ckpt(model_folder, id, output_node_names, output_graph)

    # savedModel
    # model_folder = '/Users/qiaowei/Desktop/cnn-facial-landmark/saved_model/1551857627'
    # output_node_names = 'logits/BiasAdd'
    # output_graph = './faceFrozenSrc.pb'
    #
    # freeze_graph(model_folder, output_node_names, output_graph)

if __name__ == '__main__':
    main()
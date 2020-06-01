#coding:utf-8#
import tensorflow as tf
import numpy as np
import cv2 as cv
import os

def checkGraphByCkpt():
    model_base = 'C:/Users/qiaowei6/Desktop\ssdlite_mobilenet_v2_coco_2018_05_09/ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt'

    input_checkpoint = model_base
    log_path = './ssdliteCkpt/'

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        # output_graph_def = tf.graph_util.convert_variables_to_constants(
        #     sess,
        #     input_graph_def,
        #     'logits/bias'.split(',')
        # )

        summary_writer = tf.summary.FileWriter(log_path, sess.graph)

def checkGraphByFrozenPb():
    model_path = "D:\PyProject\MMNet\output\model.pb"
    log_path = './'

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    cpn_sess = tf.Session()
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        cpn_sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    # cpn_sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(log_path, cpn_sess.graph)

def checkGraphBySavedPb():
    model_path = "/Users/qiaowei/Desktop/cnn-facial-landmark/saved_model/1551841381"
    log_path = './logPbFaceSM/'
    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess, ["serve"], model_path)
      graph = tf.get_default_graph()

      summary_writer = tf.summary.FileWriter(log_path, graph)

def load_cpn_graph(model_path):
    cpn_sess = tf.Session()
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        cpn_sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图


    cpn_inputs = cpn_sess.graph.get_tensor_by_name('Placeholder:0')
    cpn_prediction = cpn_sess.graph.get_tensor_by_name('refine_out/BatchNorm/FusedBatchNorm:0')
    cpn_sess.run(tf.global_variables_initializer())

    # example
    data_shape = (384, 288)
    resize_size = (288, 384)
    output_shape = (96, 72)

    frame = cv.imread("")
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    frameDet = cv.resize(frame, resize_size)

    pred = cpn_sess.run(cpn_prediction, feed_dict={cpn_inputs: [frameDet]})

    cls_skeleton = np.zeros((17, 3))
    res = np.array(pred[0]).transpose(2, 0, 1)
    print(np.shape(res))
    # res = res.transpose(0, 3, 1, 2)

    nr_skeleton = 17
    r0 = res.copy()
    for w in range(nr_skeleton):
        res[w] /= np.amax(res[w])
    border = 10
    dr = np.zeros((nr_skeleton, output_shape[0] + 2 * border, output_shape[1] + 2 * border))
    dr[:, border:-border, border:-border] = res[:nr_skeleton].copy()

    for w in range(nr_skeleton):
        dr[w] = cv.GaussianBlur(dr[w], (21, 21), 0)

    for w in range(nr_skeleton):
        lb = dr[w].argmax()
        y, x = np.unravel_index(lb, dr[w].shape)
        dr[w, y, x] = 0
        lb = dr[w].argmax()
        py, px = np.unravel_index(lb, dr[w].shape)
        y -= border
        x -= border
        py -= border + y
        px -= border + x
        ln = (px ** 2 + py ** 2) ** 0.5
        delta = 0.25
        if ln > 1e-3:
            x += delta * px / ln
            y += delta * py / ln
        x = max(0, min(x, output_shape[1] - 1))
        y = max(0, min(y, output_shape[0] - 1))

        # cls_skeleton[w, :2] = (x * 4 + 2, y * 4 + 3)
        cls_skeleton[w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)] / 255.0
        cls_skeleton[w, 1] = cls_skeleton[w, 1] / output_shape[0] * frameHeight
        cls_skeleton[w, 0] = cls_skeleton[w, 0] / output_shape[1] * frameWidth

def main():
    checkGraphByFrozenPb()

if __name__ == '__main__':
    main()


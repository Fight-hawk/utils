import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
import time
from openvino.inference_engine import IENetwork, IECore

def loadModel(model_xml):
    xml = model_xml
    bin = os.path.splitext(xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(xml, bin))
    net = IENetwork(model=xml, weights=bin)
    ie = IECore()
    device = 'GPU'
    log.info("Device info:")
    versions = ie.get_versions(device)
    print("{}{}".format(" " * 8, device))
    print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[device].major, versions[device].minor))
    print("{}Build ........... {}".format(" " * 8, versions[device].build_number))

    # --------------------------- 3. Read and preprocess input --------------------------------------------
    input_blob = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_blob].shape
    print(n, c, h, w)
    images = np.ndarray(shape=(n, c, h, w))
    net.batch_size = 1

    log.info("Preparing input blobs")
    assert (len(net.inputs.keys()) == 1 or len(
        net.inputs.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"

    out_blob = next(iter(net.outputs))
    n, c, h, w = net.outputs[out_blob].shape
    print(n, c, h, w)

    input_info = net.inputs[next(iter(net.inputs.keys()))]
    output_info = net.outputs[next(iter(net.outputs.keys()))]
    input_info.precision = "FP32"
    output_info.precision = "FP32"

    log.info("Loading model to the device")
    exec_net = ie.load_network(network=net, device_name=device)
    log.info("Creating infer request and starting inference")

    return images, exec_net, input_blob, out_blob

INPUT_SIZE = 256

XDF = cv2.imread("san.jpg")

espnet_path = "op11.xml"

images, exec_net, input_blob, out_blob = loadModel(espnet_path)

cap = cv2.VideoCapture(0)
ret, img = cap.read()

def seg_process(image):
    # opencv
    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)

    image_resize = (image_resize - (104., 112., 121.,)) / 255.0
    image_resize = np.transpose(image_resize, (2, 0, 1))
    # image_resize = np.expand_dims(image_resize, axis=0)

    images[0] = image_resize

    xdf = cv2.resize(XDF, dsize=(origin_w, origin_h))
    # -----------------------------------------------------------------

    alpha = exec_net.infer(inputs={input_blob: images})
    alpha = alpha[out_blob]
    alpha_np = alpha[0, 0, :, :]
    fg_alpha = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    # _, fg_alpha = cv2.threshold(fg_alpha, 0.5, 1.0, cv2.THRESH_TOZERO)
    bg_alpha = 1.0 - fg_alpha

    # -----------------------------------------------------------------
    # fg = fg_alpha * 255
    fg = np.multiply(fg_alpha[..., np.newaxis], image)
    bg = np.multiply(bg_alpha[..., np.newaxis], xdf)

    # gray
    # bg = image
    # bg_alpha = 1 - fg_alpha[..., np.newaxis]
    # bg_alpha[bg_alpha < 0] = 0
    #
    # bg_gray = np.multiply(bg_alpha, image)
    # bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)
    #
    # bg[:, :, 0] = bg_gray
    # bg[:, :, 1] = bg_gray
    # bg[:, :, 2] = bg_gray

    # -----------------------------------------------------------------
    # fg : color, bg : gray
    out = fg + bg
    # fg : color
    # out = fg_alpha * 255
    out = out.astype(np.uint8)

    return out

while ret:

    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()
    result = seg_process(frame)

    cv2.imshow("test", result)

    waitTime =time.time() - t0

    print(waitTime)
    if waitTime < 2:
        waitTime = 1


    if cv2.waitKey(int(waitTime)) & 0xFF == ord('q'):
        break

cap.release()

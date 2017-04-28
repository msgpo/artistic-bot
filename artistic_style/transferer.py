__all__ = ('transfer_style',)

import tensorflow as tf
import numpy as np
import scipy.misc

import artistic_style.transform_net as transform_net


MAX_SIZE = 720


def gray2rgb(img):
    height, width = img.shape
    out = np.zeros((height, width, 3))
    out[:, :, 0] = img
    out[:, :, 1] = img
    out[:, :, 2] = img
    return out


def scale_image(image):
    height, width, _ = image.shape
    size = max(height, width)
    if size < MAX_SIZE:
        return image
    scale_ratio = MAX_SIZE / size
    scaled = scipy.misc.imresize(image, scale_ratio).astype(np.float32)
    return scaled


def preprocess_image(image):
    out = scale_image(image)
    return out


def transfer_style(content, model_filename, device='/cpu:0'):
    if len(content.shape) == 2:
        content = gray2rgb(content)
    content = preprocess_image(content)

    with tf.Graph().as_default(), tf.device(device):
        content_batch = tf.expand_dims(content, 0)
        stylized = transform_net.build_net(content_batch)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_filename)

            out = sess.run(stylized)
            out = np.squeeze(out, 0)
    return out

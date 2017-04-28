import numpy as np
import tensorflow as tf


MEAN_PIXEL = np.array([123.68, 116.779, 103.939], dtype=np.float32)
LAYERS = (
    'conv1_1', 'conv1_2', 'pool1',
    'conv2_1', 'conv2_2', 'pool2',
    'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'pool3',
    'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4',
    'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5'
)


def load_weights(path):
    weights = np.load(path, encoding='latin1').item()
    return weights


def preprocess(image, pack=True):
    mean = tf.constant(MEAN_PIXEL, name='mean_pixel')
    out = image - mean
    if pack:
        out = tf.expand_dims(out, 0)
    return out


def build_net(value, weights, name=None):
    net = {}
    if not name:
        name = 'loss_net'

    with tf.name_scope(name):
        bottom = value
        for name in LAYERS:
            kind = name[:4]
            if kind == 'conv':
                curr = conv_layer(bottom, weights[name], name)
            elif kind == 'pool':
                curr = pool_layer(bottom, name)
            else:
                raise RuntimeError('unknown layer kind')
            bottom, net[name] = curr, curr
    return net


def conv_layer(value, weights, name):
    strides = (1, 1, 1, 1)
    with tf.name_scope(name):
        filter = tf.constant(weights[0], name='filter')
        bias = tf.constant(weights[1], name='bias')
        conv = tf.nn.conv2d(value, filter, strides, padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
    return relu


def pool_layer(value, name):
    kernel = (1, 2, 2, 1)
    strides = (1, 2, 2, 1)
    return tf.nn.max_pool(value, kernel, strides,
                          padding='SAME', name=name)

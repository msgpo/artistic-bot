import tensorflow as tf


WEIGHTS_STDDEV = 0.1


def build_net(value, name=None):
    if not name:
        name = 'transform_net'

    with tf.name_scope(name):
        conv1 = conv_layer(
            value,
            filter_size=9,
            in_channels=3,
            out_channels=32,
            stride=1,
            name='conv1')

        conv2 = conv_layer(
            conv1,
            filter_size=3,
            in_channels=32,
            out_channels=64,
            stride=2,
            name='conv2')

        conv3 = conv_layer(
            conv2,
            filter_size=3,
            in_channels=64,
            out_channels=128,
            stride=2,
            name='conv3')

        residual1 = residual_block(
            conv3,
            num_filters=128,
            name='residual1')

        residual2 = residual_block(
            residual1,
            num_filters=128,
            name='residual2')

        residual3 = residual_block(
            residual2,
            num_filters=128,
            name='residual3')

        residual4 = residual_block(
            residual3,
            num_filters=128,
            name='residual4')

        residual5 = residual_block(
            residual4,
            num_filters=128,
            name='residual5')

        deconv1 = resize_conv_layer(
            residual5,
            filter_size=3,
            in_channels=128,
            out_channels=64,
            stride=2,
            name='deconv1')

        deconv2 = resize_conv_layer(
            deconv1,
            filter_size=3,
            in_channels=64,
            out_channels=32,
            stride=2,
            name='deconv2')

        conv4 = conv_layer(
            deconv2,
            filter_size=9,
            in_channels=32,
            out_channels=3,
            stride=1,
            relu=False,
            name='conv4')

        scale = tf.constant(127.5)
        shift = tf.constant(127.5)
        out = scale * tf.tanh(conv4) + shift
    return out


def conv_layer(value, filter_size, in_channels, out_channels,
               stride, name, relu=True):
    filter_shape = (filter_size, filter_size, in_channels, out_channels)
    strides = (1, stride, stride, 1)
    paddings = ((0, 0),
                (filter_size // 2, filter_size // 2),
                (filter_size // 2, filter_size // 2),
                (0, 0))

    with tf.name_scope(name):
        weights = tf.truncated_normal(filter_shape, stddev=WEIGHTS_STDDEV)
        filter = tf.Variable(weights, dtype=tf.float32, name='filter')
        padded = tf.pad(value, paddings, mode='REFLECT')
        out = tf.nn.conv2d(padded, filter, strides, padding='VALID')
        out = instance_norm(out, out_channels, 'instance_norm')
        if relu:
            out = tf.nn.relu(out)
    return out


def tensor_shape(tensor):
    shape = tensor.get_shape()
    return tuple(map(lambda dim: dim.value, shape))


def resize_conv_layer(value, filter_size, in_channels, out_channels,
                      stride, name):
    """Make a deconvolution layer.

    Links:
        http://distill.pub/2016/deconv-checkerboard/
    """
    value_shape = tensor_shape(value)
    height, width = value_shape[1], value_shape[2]
    new_height, new_width = int(height * stride), int(width * stride)
    scaled_size = (new_height, new_width)

    with tf.name_scope(name):
        out = tf.image.resize_images(value, scaled_size,
                                     tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        out = conv_layer(out, filter_size, in_channels,
                         out_channels, stride=1, name='conv')
    return out


def residual_block(value, num_filters, name, filter_size=3):
    stride = 1
    with tf.name_scope(name):
        residual = conv_layer(value, filter_size, num_filters,
                              num_filters, stride, name='conv_1')
        residual = conv_layer(residual, filter_size, num_filters,
                              num_filters, stride, relu=False, name='conv_2')
        out = value + residual
    return out


def instance_norm(value, in_channels, name, eps=1e-8):
    with tf.name_scope(name):
        mean, var = tf.nn.moments(value, (1, 2), keep_dims=True)
        scale = tf.Variable(tf.ones([in_channels]), name='scale')
        shift = tf.Variable(tf.zeros([in_channels]), name='shift')
        out = tf.nn.batch_normalization(value, mean, var, shift, scale, eps)
        return out

import numpy as np
import tensorflow as tf
import os
import sys
import argparse

from contextlib import contextmanager
from datetime import datetime

from artistic_style import imread
from artistic_style import create_dir
from artistic_style import transform_net
from artistic_style import loss_net


CONTENT_LAYERS = ('conv3_3',)
STYLE_LAYERS = ('conv1_2', 'conv2_1', 'conv3_1', 'conv4_1')

DIRNAME = os.path.abspath(os.path.dirname(__file__))
CHECKPOINTS_DIR = os.path.join(DIRNAME, 'checkpoints')
SUMMARY_DIR = '/tmp/tensorflow/fast-style'
NUM_EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 1e2
DEVICE = '/cpu:0'
SUMMARY_PERIOD = 10
CHECKPOINT_PERIOD = 1000


@contextmanager
def job_timer(info):
    tf.logging.info(info)
    sys.stdout.flush()
    start = datetime.now()
    yield
    delta = datetime.now() - start
    tf.logging.info('%s DONE (took %.2fs)', info, delta.total_seconds())


def progress(message):
    print(message, end='')
    sys.stdout.flush()


def build_gram(layer):
    shape = tf.shape(layer)
    num_batches, height, width, num_filters = (shape[i] for i in range(4))
    size = height * width * num_filters
    f = tf.reshape(layer, (num_batches, height * width, num_filters))
    f_T = tf.transpose(f, perm=(0, 2, 1))
    gram = tf.matmul(f_T, f) / tf.to_float(size)
    return gram


def compute_content_features(content, weights, device):
    """Compute content features."""
    content_features = {}
    with tf.name_scope('content_features'):
        content_pre = loss_net.preprocess(content, pack=False)
        net = loss_net.build_net(content_pre, weights)
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer]
    return content_features


def compute_style_features(image, weights, device):
    """Compute style features."""
    with tf.Graph().as_default(), tf.device(device):
        image_pre = loss_net.preprocess(image)
        net = loss_net.build_net(image_pre, weights)
        fetches = {layer_name:build_gram(net[layer_name])
                   for layer_name in STYLE_LAYERS}
        with tf.Session() as sess:
            style_features = sess.run(fetches)
    return style_features


def compute_content_loss(net, content_features):
    """Compute content loss."""
    loss = 0
    with tf.name_scope('content_loss'):
        for layer in CONTENT_LAYERS:
            size = tf.size(net[layer])
            loss += (tf.nn.l2_loss(net[layer] - content_features[layer]) /
                tf.to_float(size))
    return loss


def compute_style_loss(net, style_features):
    """Compute style loss."""
    loss = 0
    style_layer_weight = 1 / len(style_features)
    with tf.name_scope('style_loss'):
        for layer_name, style_gram in style_features.items():
            layer = net[layer_name]
            size = tf.size(layer)
            gram = build_gram(layer)
            loss += (style_layer_weight *
                tf.nn.l2_loss(gram - style_gram) / tf.to_float(size))
    return loss


def read_images(filename_queue, num_channels):
    reader = tf.TFRecordReader()
    _, examples = reader.read(filename_queue)
    features = tf.parse_single_example(
        examples,
        features={'data': tf.FixedLenFeature([], tf.string)})
    images = tf.image.decode_jpeg(features['data'], num_channels)
    return images


def build_input_pipeline(contents_dir, num_epochs, batch_shape):
    batch_size, height, width, num_channels = batch_shape
    with tf.name_scope('input_pipeline'), tf.device('/cpu:0'):
        filenames = tf.train.match_filenames_once(
            os.path.join(contents_dir, '*.tfrecords'),
            name='filenames')
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs,
            name='filename_queue', capacity=128)
        images = read_images(filename_queue, num_channels)
        images = tf.image.resize_images(images, (height, width))
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        batch_images = tf.train.shuffle_batch(
            [images], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    return batch_images


def train_transform_net(style, style_name, options):
    contents_dir = options.contents
    checkpoints_dir = options.checkpoints
    vgg_weights_file = options.weights
    num_epochs = options.num_epochs
    batch_size = options.batch_size
    learning_rate = options.learning_rate
    beta1 = options.beta1
    beta2 = options.beta2
    eps = options.epsilon
    content_weight = options.content_weight
    style_weight = options.style_weight
    device = options.device
    verbose = options.verbose
    summary_dir = options.summary
    summary_period = options.summary_period
    checkpoint_period = options.checkpoint_period
    train_from = options.train_from

    create_dir(checkpoints_dir)
    model_weights_file = os.path.join(checkpoints_dir, style_name)

    batch_shape = (batch_size, 256, 256, 3)

    with job_timer('Loading VGG weights...'):
        weights = loss_net.load_weights(vgg_weights_file)
    with job_timer('Computing style features...'):
        style_features = compute_style_features(style, weights, device)

    tf.logging.info('batch shape = %s', str(batch_shape))
    tf.logging.info('content_weight = %.3f', content_weight)
    tf.logging.info('style_weight = %.3f', style_weight)
    tf.logging.info('learning_rate = %.3f', learning_rate)
    tf.logging.info('beta1 = %.3f', beta1)
    tf.logging.info('beta2 = %.3f', beta2)
    tf.logging.info('epsilon = %f', eps)

    with tf.Graph().as_default(), tf.device(device):
        batch_op = build_input_pipeline(
            contents_dir, num_epochs, batch_shape)

        content = tf.placeholder(tf.float32, shape=batch_shape)

        # Stick together transform net and perceptual loss net.
        transformed = transform_net.build_net(content)
        transformed_pre = loss_net.preprocess(transformed, pack=False)
        net = loss_net.build_net(transformed_pre, weights)

        with tf.name_scope('loss'):
            content_features = compute_content_features(
                content, weights, device)
            content_loss = compute_content_loss(net, content_features)
            style_loss = compute_style_loss(net, style_features)
            loss = (content_weight * content_loss +
                    style_weight * style_loss)
            tf.summary.scalar('content_loss', content_loss)
            tf.summary.scalar('style_loss', style_loss)
            tf.summary.scalar('total_loss', loss)

        summary_op = tf.summary.merge_all()

        # Save only transform_net's weights.
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='transform_net')
        saver = tf.train.Saver(train_vars)
        writer = tf.summary.FileWriter(summary_dir)

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, eps)
        train_step = optimizer.minimize(loss)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        if verbose:
            config.log_device_placement = True

        with tf.Session(config=config) as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            if train_from:
                tf.logging.info('Restoring model from %r' % train_from)
                saver.restore(sess, train_from)
            writer.add_graph(sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                step = 0
                while not coord.should_stop():
                    progress('\r:: [INFO] step = %d' % step)
                    content_batch = sess.run(batch_op)
                    feed_dict = {content: content_batch}
                    if step % summary_period == 0:
                        _, summary = sess.run([train_step, summary_op],
                                              feed_dict=feed_dict)
                        writer.add_summary(summary, step)
                    else:
                        sess.run(train_step, feed_dict=feed_dict)

                    if step % checkpoint_period == 0:
                        saver.save(sess, model_weights_file,
                                   global_step=step)

                    step += 1
            except tf.errors.OutOfRangeError:
                pass
            finally:
                model_weights_file = os.path.join(
                    checkpoints_dir, style_name + '-final')
                saver.save(sess, model_weights_file)
                coord.request_stop()

            coord.join(threads)
            writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contents',
                        help='directory with content targets', required=True)
    parser.add_argument('--weights',
                        help='VGG weights file', required=True)
    parser.add_argument('--checkpoints',
                        help='directory where checkpoints will be stored',
                        default=CHECKPOINTS_DIR)
    parser.add_argument('--checkpoint-period', type=int,
                        help='save model every n steps',
                        default=CHECKPOINT_PERIOD)
    parser.add_argument('--style',
                        help='style file', required=True)
    parser.add_argument('--style-name',
                        help='name of the style', required=True)
    parser.add_argument('--num-epochs',
                        help='set number of iterations', type=int,
                        default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int,
                        help='set batch size', default=BATCH_SIZE)
    parser.add_argument('--device', help='compute device',
                        default=DEVICE)
    parser.add_argument('--lr',
                        help='set learning rate', type=float,
                        default=LEARNING_RATE, dest='learning_rate')
    parser.add_argument('--beta1',
                        help='exp decay rate for the 1st moment estimates',
                        default=BETA1, type=float)
    parser.add_argument('--beta2',
                        help='exp decay rate for the 2nd moment estimates',
                        default=BETA2, type=float)
    parser.add_argument('--epsilon',
                        help='small constant for numerical stability',
                        default=EPSILON, type=float)
    parser.add_argument('--cw',
                        help='set content weight', type=float,
                        default=CONTENT_WEIGHT, dest='content_weight')
    parser.add_argument('--sw',
                        help='set style weight', type=float,
                        default=STYLE_WEIGHT, dest='style_weight')
    parser.add_argument('--verbose',
                        help='display device placement, etc.',
                        action='store_true')
    parser.add_argument('--summary',
                        help='set path to TensorBoard log directory',
                        default=SUMMARY_DIR)
    parser.add_argument('--summary-period', type=int,
                        help='add new summary every n steps',
                        default=SUMMARY_PERIOD)
    parser.add_argument('--train-from',
                        help='Start training from given model')
    args = parser.parse_args()
    return args


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    style = imread(args.style)
    train_transform_net(style, args.style_name, args)


if __name__ == '__main__':
    main()

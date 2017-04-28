import tensorflow as tf
import numpy as np
import argparse

from artistic_style import imread
from artistic_style import imsave
from artistic_style import transfer_style


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('content',
                        help='image to be transformed')
    parser.add_argument('-o', '--output',
                        help='output file',
                        required=True)
    parser.add_argument('-m', '--model',
                        help='pretrained model',
                        required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    content_filename = args.content
    output_filename = args.output
    model_weights_filename = args.model

    content = imread(content_filename)
    out = transfer_style(content, model_weights_filename)
    imsave(output_filename, out)


if __name__ == '__main__':
    main()

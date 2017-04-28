__all__ = ('imread',
           'imsave',
           'create_dir')

import numpy as np
import scipy.misc
import os


def imread(path):
    img = scipy.misc.imread(path)
    img = img.astype(np.float32)
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

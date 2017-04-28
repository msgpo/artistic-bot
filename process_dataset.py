import tensorflow as tf
import argparse
import os
import itertools
import concurrent.futures

from tqdm import tqdm
from glob import iglob
from concurrent.futures import ThreadPoolExecutor
from scipy.misc import imread


NUM_WORKERS = 2
RECORD_FILE_FORMAT = 'train-%04d.tfrecords'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',
                        help='dataset directory',
                        required=True)
    parser.add_argument('--output-dir',
                        help='output directory',
                        required=True)
    parser.add_argument('--bucket-size', type=int,
                        help='number of items in a bucket',
                        required=True)
    parser.add_argument('--num-workers', type=int,
                        help='number of threads',
                        default=NUM_WORKERS)
    args = parser.parse_args()
    return args


def _bytes_features(value):
    bytes_list = tf.train.BytesList(value=[value])
    return tf.train.Feature(bytes_list=bytes_list)


def _int_features(value):
    int64_list = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=int64_list)


def worker(files, output_dir, bucket_id):
    """Convert images to TF format.

    Args:
        files (list): List of image files.
        output_dir (str): Path to directory where records will be saved.
        bucket_id (int): id of a bucket.

    Returns:
        str: Path to the result tfrecords file.
    """
    record_file = os.path.join(output_dir, RECORD_FILE_FORMAT % bucket_id)
    writer = tf.python_io.TFRecordWriter(record_file)
    for filename in files:
        with open(filename, 'rb') as f:
            data = f.read()
        features = tf.train.Features(feature={
            'data': _bytes_features(tf.compat.as_bytes(data))
        })
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()
    return record_file


def group(iterable, n):
    """Group data into n-length chunks.

    Args:
        iterable (iterable): Data.
        n (int): Length of each chunk.

    Yields:
        iterator: Chunks of length n.
    """
    it = iter(iterable)
    while True:
        chunk = itertools.islice(it, n)
        try:
            head = next(chunk)
        except StopIteration:
            return
        yield itertools.chain((head,), chunk)


def list_images(path):
    """List image files in a directory.

    Args:
        path (str): Path to directory.

    Returns:
        iterator: List of images.
    """
    exts = ('*.jpg', '*.png')
    images = itertools.chain.from_iterable(
        iglob(os.path.join(path, ext)) for ext in exts)
    return images


def main():
    args = parse_args()

    images = list_images(args.dataset_dir)
    buckets = group(images, args.bucket_size)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        tasks = []
        for bucket_id, files in enumerate(buckets):
            worker_args = (tuple(files), args.output_dir, bucket_id)
            future = executor.submit(worker, *worker_args)
            tasks.append(future)

        completed_tasks = concurrent.futures.as_completed(tasks)
        for future in tqdm(completed_tasks, total=len(tasks)):
            record_file = future.result()
            tqdm.write(':: [DONE] ' + record_file)


if __name__ == '__main__':
    main()

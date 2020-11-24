"""dataset.py
This module implements functions for reading ImageNet (ILSVRC2012)
dataset in TFRecords format.

Cleaned up for Tensorflow 2.0
"""

import os
import tensorflow as tf
import re

#No augmentation during training
def _parse_fn(example_serialized, image_size = 299):
  """Helper function for parse_fn_train() and parse_fn_valid()
    Each Example proto (TFRecord) contains the following fields:
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>
    Args:
        example_serialized: scalar Tensor tf.string containing a
        serialized Example protocol buffer.
    Returns:
        image_buffer: Tensor tf.string containing the contents of
        a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
  """
  feature_map = {
        'image/filename': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                   default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                  default_value=''),
  }
  parsed = tf.io.parse_single_example(example_serialized, feature_map)
  image = tf.image.decode_jpeg(parsed['image/encoded'])
  image = tf.image.resize(image, [image_size, image_size])
  image = tf.cast(image, tf.float32) / 255.0
  label = tf.one_hot(parsed['image/class/label'], 3, dtype=tf.float32)
  return image, label

def _parse_fn_test(example_serialized, image_size = 299):
  """Helper function for parse_fn_train() and parse_fn_valid()
    Each Example proto (TFRecord) contains the following fields:
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>
    Args:
        example_serialized: scalar Tensor tf.string containing a
        serialized Example protocol buffer.
    Returns:
        image_buffer: Tensor tf.string containing the contents of
        a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
  """
  feature_map = {
        'image/filename': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                   default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                  default_value=''),
  }
  parsed = tf.io.parse_single_example(example_serialized, feature_map)
  image = tf.image.decode_jpeg(parsed['image/encoded'])
  image = tf.image.resize(image, [image_size, image_size])
  image = tf.cast(image, tf.float32) / 255.0
  label = tf.one_hot(parsed['image/class/label'], 3, dtype=tf.float32)
  filename = tf.cast(parsed['image/filename'],tf.string)
  return image, label, filename

def _parse_fn_predict(example_serialized, image_size = 299):
  """Helper function for parse_fn_train() and parse_fn_valid()
    Each Example proto (TFRecord) contains the following fields:
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>
    Args:
        example_serialized: scalar Tensor tf.string containing a
        serialized Example protocol buffer.
    Returns:
        image_buffer: Tensor tf.string containing the contents of
        a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
  """
  feature_map = {
        'image/filename': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
  }
  parsed = tf.io.parse_single_example(example_serialized, feature_map)
  image = tf.image.decode_jpeg(parsed['image/encoded'])
  image = tf.image.resize(image, [image_size, image_size])
  image = tf.cast(image, tf.float32) / 255.0
    #label = tf.one_hot(parsed['image/class/label'], 3, dtype=tf.float32)
  filename = tf.cast(parsed['image/filename'],tf.string)
  return image, filename



# OBSOLETE with Tensorflow 2.0
# def get_dataset(tfrecords_dir, subset, batch_size):
#     """Read TFRecords files and turn them into a TFRecordDataset."""
#     files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
#     shards = tf.data.Dataset.from_tensor_slices(files)
#     shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
#     shards = shards.repeat()
#     dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
#     dataset = dataset.shuffle(buffer_size=512)
#     parser = _parse_fn
#     dataset = dataset.apply(
#         tf.data.experimental.map_and_batch(
#             map_func=parser,
#             batch_size=batch_size,
#             num_parallel_calls=4))
#     dataset = dataset.prefetch(batch_size)
#     return dataset


def count_dataset(tfrecords_dir, subset, batch_size, epochs):
  """Read TFRecords files and turn them into a TFRecordDataset."""
  files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
  #shards = tf.data.Dataset.from_tensor_slices(files)
  #shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
  #dataset = shards.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  shards = tf.data.Dataset.from_tensor_slices(files)
  dataset = tf.data.TFRecordDataset(shards, num_parallel_reads=tf.data.experimental.AUTOTUNE)
  tmpdataset = dataset.batch(batch_size, drop_remainder = False)
  count = tmpdataset.reduce(0, lambda x, _: x + 1).numpy()

  #The code below will be removed if not needed
  #tmpdataset = dataset.batch(batch_size, drop_remainder = False)
  #count = tmpdataset.reduce(0, lambda x, _: x + 1).numpy()
  return count

def get_dataset(tfrecords_dir, subset, batch_size, epochs):
  """Read TFRecords files and turn them into a TFRecordDataset."""
  files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
  shards = tf.data.Dataset.from_tensor_slices(files)
  shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
  dataset = shards.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.shuffle(buffer_size = 512)
  dataset = dataset.map(_parse_fn,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    #dataset = dataset.cache()
  dataset = dataset.batch(batch_size, drop_remainder = False)
  dataset = dataset.repeat(epochs)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset

def get_one_slide(tfrecords_dir, file, batch_size = 1, evaluate = True):
  """Read TFRecord of one slide at a time"""
  dataset = tf.data.TFRecordDataset([os.path.join(tfrecords_dir,file)])
  if evaluate:
    datasets = dataset.map(_parse_fn_test)
  else:
    datasets = dataset.map(_parse_fn_predict)
  batched_datasets = datasets.batch(batch_size, drop_remainder=False)
  batched_datasets = batched_datasets.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return batched_datasets



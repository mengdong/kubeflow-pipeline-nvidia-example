#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os

import tensorflow as tf
import horovod.tensorflow as hvd

from utils import hvd_utils

__all__ = ["get_synth_input_fn", "normalized_inputs"]

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

_NUM_CHANNELS = 3
_NUM_DATA_FILES = 5

NUM_IMAGES = {
  'train': 50000,
  'validation': 10000,
}

def get_synth_input_fn(batch_size, height, width, num_channels, data_format, num_classes, dtype=tf.float32):
  """Returns an input function that returns a dataset with random data.

  This input_fn returns a data set that iterates over a set of random data and
  bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
  copy is still included. This used to find the upper throughput bound when
  tunning the full input pipeline.

  Args:
      height: Integer height that will be used to create a fake image tensor.
      width: Integer width that will be used to create a fake image tensor.
      num_channels: Integer depth that will be used to create a fake image tensor.
      num_classes: Number of classes that should be represented in the fake labels
          tensor
      dtype: Data type for features/images.

  Returns:
      An input_fn that can be used in place of a real one to return a dataset
      that can be used for iteration.
  """

  if data_format not in ["NHWC", "NCHW"]:
    raise ValueError("Unknown data_format: %s" % str(data_format))

  if data_format == "NHWC":
    input_shape = [batch_size, height, width, num_channels]
  else:
    input_shape = [batch_size, num_channels, height, width]

  # Convert the inputs to a Dataset.
  inputs = tf.truncated_normal(input_shape, dtype=dtype, mean=127, stddev=60, name='synthetic_inputs')
  labels = tf.random_uniform([batch_size], minval=0, maxval=num_classes - 1, dtype=tf.int32, name='synthetic_labels')

  data = tf.data.Dataset.from_tensors((inputs, labels))

  data = data.repeat()

  data = data.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return data

def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  assert tf.io.gfile.exists(data_dir), (
    'Run cifar10_download_and_extract.py first to download and extract the '
    'CIFAR-10 data.')

  if is_training:
    return [
      os.path.join(data_dir, 'data_batch_%d.bin' % i)
      for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]

def preprocess_image(image, is_training, height, width):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
      image, height + 8, width + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [height, width, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image

def parse_record(raw_record, is_training, dtype, height, width, _RECORD_BYTES):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.io.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [_NUM_CHANNELS, height, width])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(a=depth_major, perm=[1, 2, 0]), tf.float32)

  image = preprocess_image(image, is_training, height, width)
  image = tf.cast(image, dtype)

  return image, label

def get_tfrecords_input_fn(data_dir, num_epochs,
                           batch_size, height, width,
                           training,
                           # distort_color,
                           datasets_num_private_threads):
  # deterministic):
  """Input function which provides batches for train or eval.
  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    num_parallel_batches: Number of parallel batches for tf.data.
    parse_record_fn: Function to use for parsing the records.
    input_context: A `tf.distribute.InputContext` object passed in by
      `tf.distribute.Strategy`.
  Returns:
    A dataset that can be used for iteration.
  """

  _DEFAULT_IMAGE_BYTES = height * width * _NUM_CHANNELS
  # The record is the image plus a one-byte label
  _RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
  filenames = get_filenames(training, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

  dtype = tf.float32
  num_parallel_batches = 1

  parse_record_fn = parse_record
  # input_context = None
  #
  # if input_context:
  #   tf.compat.v1.logging.info(
  #     'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
  #       input_context.input_pipeline_id, input_context.num_input_pipelines))
  #   dataset = dataset.shard(input_context.num_input_pipelines,
  #                           input_context.input_pipeline_id)

  return process_record_dataset(
    dataset=dataset,
    is_training=training,
    batch_size=batch_size,
    height=height,
    width=width,
    _RECORD_BYTES=_RECORD_BYTES,
    shuffle_buffer=NUM_IMAGES['train'],
    parse_record_fn=parse_record_fn,
    num_epochs=num_epochs,
    dtype=dtype,
    datasets_num_private_threads=datasets_num_private_threads,
    num_parallel_batches=num_parallel_batches)


def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           height,
                           width,
                           _RECORD_BYTES,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           num_parallel_batches=1):
  """Given a Dataset with raw records, return an iterator over the records.
  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features.
    datasets_num_private_threads: Number of threads for a private
      threadpool created for all datasets computation.
    num_parallel_batches: Number of parallel batches for tf.data.
  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  # Defines a specific size thread pool for tf.data operations.
  if datasets_num_private_threads:
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = (
      datasets_num_private_threads)
    dataset = dataset.with_options(options)
    tf.compat.v1.logging.info('datasets_num_private_threads: %s',
                              datasets_num_private_threads)

  # Disable intra-op parallelism to optimize for throughput instead of latency.
  options = tf.data.Options()
  options.experimental_threading.max_intra_op_parallelism = 1
  dataset = dataset.with_options(options)

  # Prefetches a batch at a time to smooth out the time taken to load input
  # files for shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # Repeats the dataset for the number of epochs to train.
  dataset = dataset.repeat(num_epochs)

  # Parses the raw records into images and labels.
  dataset = dataset.map(
    lambda value: parse_record_fn(value, is_training, tf.float32, height, width, _RECORD_BYTES),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=False)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return dataset


def normalized_inputs(inputs):
  num_channels = inputs.get_shape()[-1]

  if inputs.get_shape().ndims != 4:
    raise ValueError('Input must be of size [batch_size, height, width, C>0]')

  if len(_CHANNEL_MEANS) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  means_per_channel = tf.reshape(_CHANNEL_MEANS, [1, 1, num_channels])
  means_per_channel = tf.cast(means_per_channel, dtype=inputs.dtype)

  inputs = tf.subtract(inputs, means_per_channel)

  return tf.divide(inputs, 255.0)

def build_serving_input_receiver_fn(shape, dtype=tf.float32,
                                          batch_size=None):
  def serving_input_receiver_fn():
      features = tf.placeholder(
          dtype=dtype, shape=[batch_size] + shape, name='input_tensor')
      return tf.estimator.export.TensorServingInputReceiver(
          features=features, receiver_tensors=features)

  return serving_input_receiver_fn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import os

from scipy import misc

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 17


def read_image(filename_queue):
  """Reads and parses examples from CIFAR100 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  result.key, label = tf.decode_csv(filename_queue.dequeue(), [[""], [""]], " ")

  # Extract raw PNG data as a string
  raw_contents = tf.read_file(result.key)

  # Decode raw data as a PNG. Defaults to uint8 encoding.
  result.uint8image = tf.image.decode_png(raw_contents)

  # TENSORFLOW BUG: image shape not statically determined, so force
  # it to have correct CIFAR100 dimensions
  result.uint8image.set_shape([32, 32, 3])

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(tf.string_to_number(label), tf.int32)

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(filename, data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  with open(filename) as f:
    delimmed = f.readlines()
  delimmed = [l.strip('\n') for l in delimmed]

  # Create a queue that produces the filename, label pairs to read.
  delimmed_queue = tf.train.string_input_producer(delimmed)

  # Read examples from files in the filename queue.
  read_input = read_image(delimmed_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)


def testing_inputs(filename, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  with open(filename) as f:
    delimmed = f.readlines()
  delimmed = [l.strip('\n') for l in delimmed]

  # Create a queue that produces the filename, label pairs to read.
  delimmed_queue = tf.train.string_input_producer(delimmed)

  # Read examples from files in the filename queue.
  read_input = read_image(delimmed_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)


def read_image_unlabeled(filename_queue):
  class StatefarmRecord(object):
    pass
  result = StatefarmRecord()

  # Read a record, getting filenames from the filename_queue.  
  result.key, _ = tf.decode_csv(filename_queue.dequeue(), [[""], [""]], " ")

  # Extract raw JPG data as a string
  raw_contents = tf.read_file(result.key)

  # Decode raw data as a PNG. Defaults to uint8 encoding.
  result.uint8image = tf.image.decode_png(raw_contents)

  # TENSORFLOW BUG: image shape not statically determined, so force
  # it to have correct CIFAR100 dimensions
  result.uint8image.set_shape((32, 32, 3))

  # Kind of hacky, but set a label so we can use the same structure
  # THIS SHOULD ALWAYS BE IGNORED DURING COMPUTATION, since we are
  # dealing with unlabaled data
  result.label = tf.cast(tf.string_to_number("0"), tf.int32)

  return result

def _generate_image_and_filename_batch(image, filename, min_queue_examples,
                                    batch_size):
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + filenames from the example queue.
  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch(
      [image, filename],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def unlabeled_inputs(filename, data_dir, batch_size):
  with open(filename) as f:
    delimmed = f.readlines()
  delimmed = [l.strip('\n') for l in delimmed]

  # Create a queue that produces the filename, label pairs to read.
  delimmed_queue = tf.train.string_input_producer(delimmed)

  # Read examples from files in the filename queue.
  read_input = read_image_unlabeled(delimmed_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_filename_batch(float_image, read_input.key,
                                         min_queue_examples, batch_size)
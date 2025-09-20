from __future__ import absolute_import, division, print_function
import os
import sys
import tensorflow as tf
from tensorflow.python.platform import gfile, flags
import numpy as np
import scipy.io as scio
from tensorflow.python.framework import ops
from PIL import Image

FLAGS = flags.FLAGS

# Global constants for image dimensions and channels
T = 1
IM_HEIGHT = 400
IM_WIDTH = 400
IM_CHANNELS = 3

def _int64_feature(value):
    """Creates an int64 Feature for TFRecord."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Creates a bytes Feature for TFRecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_images(filename_queue, new_height=None, new_width=None):
    """
    Reads and processes a single image from the filename queue.
    
    Args:
        filename_queue: A TensorFlow queue of filenames.
        new_height: (Optional) New height for resizing.
        new_width: (Optional) New width for resizing.
    
    Returns:
        A preprocessed image tensor.
    """
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)  # Use jpeg decoder; adjust if needed.
    image = tf.reshape(image, [IM_HEIGHT, IM_WIDTH, IM_CHANNELS])
    
    if new_height is not None and new_width is not None:
        image = tf.image.resize_images(image, [new_height, new_width])
    
    # Normalize by subtracting the mean values
    image = tf.cast(image, tf.float32) - np.array([104., 117., 124.])
    return image

def read_images2(filename_queue):
    """
    Reads a single image and produces two resized versions.
    
    Returns:
        image_227: Image resized to 227x227.
        image_128: Image resized to 128x128.
    """
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    image = tf.reshape(image, [IM_HEIGHT, IM_WIDTH, IM_CHANNELS])
    
    image_227 = tf.image.resize_images(image, [227, 227])
    image_227 = tf.cast(image_227, tf.float32) - np.array([104., 117., 124.])
    
    image_128 = tf.image.resize_images(image, [128, 128])
    image_128 = tf.cast(image_128, tf.float32) - np.array([104., 117., 124.])
    
    return image_227, image_128

def read_images3(input_queue):
    """
    Reads an image along with its label from a slice-input producer queue.
    
    Args:
        input_queue: A tuple (filenames, labels).
        
    Returns:
        image_227: Image resized to 227x227.
        image_128: Image resized to 128x128.
        label: The associated label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_image(file_contents, channels=3)
    image = tf.reshape(image, [IM_HEIGHT, IM_WIDTH, IM_CHANNELS])
    
    image_227 = tf.image.resize_images(image, [227, 227])
    image_227 = tf.cast(image_227, tf.float32) - np.array([104., 117., 124.])
    
    image_128 = tf.image.resize_images(image, [128, 128])
    image_128 = tf.cast(image_128, tf.float32) - np.array([104., 117., 124.])
    
    return image_227, image_128, label

def load_source_batch(filename, img_folder, batch_size, img_size, shuffle=True):
    """
    Creates a batch of preprocessed images for training.
    
    Args:
        filename: Path to the file containing image names and labels.
        img_folder: Folder where images are stored.
        batch_size: Number of images per batch.
        img_size: Size to which images are resized.
        shuffle: Whether to shuffle the filenames.
        
    Returns:
        A batch of images.
    """
    filenames = get_imgAndlabel_list(filename, img_folder)
    print('%d images to train' % (len(filenames)))
    if not filenames:
        raise RuntimeError('No data files found.')
    
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
        image = read_images(filename_queue, new_height=img_size, new_width=img_size)
        image_batch = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=4,
            capacity=1280,
            min_after_dequeue=640)
        return image_batch

def load_source_batch2(filename, img_folder, batch_size, shuffle=True):
    """
    Creates batches of images in two different sizes (227x227 and 128x128).
    
    Returns:
        Tuple: (image_227_batch, image_128_batch)
    """
    filenames = get_imgAndlabel_list(filename, img_folder)
    print('%d images to train' % (len(filenames)))
    if not filenames:
        raise RuntimeError('No data files found.')
    
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
        image_227, image_128 = read_images2(filename_queue)
        image_227_batch, image_128_batch = tf.train.shuffle_batch(
            [image_227, image_128],
            batch_size=batch_size,
            num_threads=4,
            capacity=1280,
            min_after_dequeue=640)
        return image_227_batch, image_128_batch

def load_source_batch3(filename, img_folder, batch_size, shuffle=True):
    """
    Loads a batch of images along with labels using slice_input_producer.
    
    Returns:
        Tuple: (image_227_batch, image_128_batch, label_batch)
    """
    img_list, label_list = get_imgAndlabel_list2(filename, img_folder)
    print('%d images to train' % (len(img_list)))
    
    images = ops.convert_to_tensor(img_list, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=shuffle)
    
    image_227, image_128, label = read_images3(input_queue)
    image_227_batch, image_128_batch, label_batch = tf.train.shuffle_batch(
        [image_227, image_128, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=1280,
        min_after_dequeue=640)
    return image_227_batch, image_128_batch, label_batch

def get_imgAndlabel_list(filename, img_folder):
    """
    Reads a list of image paths from a file.
    
    Each line in the file should be: "img_name label"
    
    Returns:
        List of full image paths.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    imgname_lists = []
    for line in lines:
        img_name = line.split()[0]
        imgname_lists.append(os.path.join(img_folder, img_name))
    return imgname_lists

def get_imgAndlabel_list2(filename, img_folder):
    """
    Reads a list of image paths and corresponding labels from a file.
    
    Each line in the file should be: "img_name label"
    
    Returns:
        Tuple: (list of full image paths, list of labels)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    imgname_lists = []
    label_lists = []
    for line in lines:
        img_name, label = line.split()
        imgname_lists.append(os.path.join(img_folder, img_name))
        label_lists.append(int(label))
    return imgname_lists, label_lists

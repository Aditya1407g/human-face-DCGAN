from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import cv2
import imageio

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    """
    Read and transform an image.
    
    Args:
        image_path (str): Path to the image file.
        image_size (int): Desired image size (for cropping/resizing).
        is_crop (bool): Whether to perform center cropping.
        resize_w (int): Target size after cropping.
        is_grayscale (bool): If True, read the image as grayscale.
        
    Returns:
        numpy.ndarray: Transformed image.
    """
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    """
    Save a grid of images after applying inverse transformation.
    
    Args:
        images (numpy.ndarray): Array of images.
        size (tuple): Grid dimensions (rows, cols).
        image_path (str): File path to save the merged image.
        
    Returns:
        None
    """
    return imsave(inverse_transform(images), size, image_path)

def save_images2(images, size, image_path):
    """
    Save a grid of images without applying inverse transformation.
    
    Args:
        images (numpy.ndarray): Array of images.
        size (tuple): Grid dimensions (rows, cols).
        image_path (str): File path to save the merged image.
        
    Returns:
        None
    """
    return imsave(images, size, image_path)

def imread(path, is_grayscale=False):
    """
    Read an image from the specified path.
    
    Args:
        path (str): Image file path.
        is_grayscale (bool): If True, read the image in grayscale.
        
    Returns:
        numpy.ndarray: Image array of type float.
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def save_source(images, size, path):
    """
    Save source images by merging them and adding back the mean.
    
    Args:
        images (numpy.ndarray): Array of images.
        size (tuple): Grid dimensions.
        path (str): Output file path.
        
    Returns:
        None
    """
    img = merge(images, size)
    mean = np.array([104., 117., 124.])
    img = np.uint8(img + mean)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])

def merge_images(images, size):
    """
    Merge images (alias for inverse_transform).
    
    Args:
        images (numpy.ndarray): Array of images.
        size (tuple): Grid dimensions.
        
    Returns:
        numpy.ndarray: Merged image.
    """
    return inverse_transform(images)

def merge(images, size):
    """
    Merge a set of images into a grid.
    
    Args:
        images (numpy.ndarray): Array of images with shape (num, height, width, channels).
        size (tuple): Tuple (rows, cols) defining the grid.
        
    Returns:
        numpy.ndarray: Merged image.
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    """
    Save images as a merged image file.
    
    Args:
        images (numpy.ndarray): Array of images.
        size (tuple): Grid dimensions.
        path (str): Output file path.
        
    Returns:
        None
    """
    merged_img = merge(images, size)
    return imageio.imwrite(path, (merged_img * 255).astype(np.uint8))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    """
    Center crop the image and resize it.
    
    Args:
        x (numpy.ndarray): Input image.
        crop_h (int): Height to crop.
        crop_w (int, optional): Width to crop; if None, use crop_h.
        resize_w (int): Target size after cropping.
        
    Returns:
        numpy.ndarray: Cropped and resized image.
    """
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    """
    Transform an image by optionally cropping/resizing and normalizing to [-1, 1].
    
    Args:
        image (numpy.ndarray): Input image.
        npx (int): Number of pixels for crop height.
        is_crop (bool): Whether to crop the image.
        resize_w (int): Resize width.
        
    Returns:
        numpy.ndarray: Transformed image.
    """
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.

def inverse_transform(images):
    """
    Inverse transform the image values from [-1, 1] back to [0, 1].
    
    Args:
        images (numpy.ndarray): Array of images.
        
    Returns:
        numpy.ndarray: Inverse transformed images.
    """
    return (images + 1.) / 2.

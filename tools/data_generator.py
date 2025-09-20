import numpy as np
import cv2
import random
from PIL import Image
import os

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
"""

class ImageDataGenerator:
    def __init__(self, batch_size, height, width, z_dim, shuffle=True,
                 scale_size=(64, 64), classes=5, mode='train'):
        """
        Initialize the image data generator.
        
        Args:
            batch_size (int): Number of samples per batch.
            height (int): Image height.
            width (int): Image width.
            z_dim (int): Dimensionality of noise vector.
            shuffle (bool): Whether to shuffle the image list.
            scale_size (tuple): The size to which images are resized.
            classes (int): Number of classes.
            mode (str): 'train' or 'test' mode.
        """
        self.root_folder = 'E:/takeoff/AgeCGan/checkpoints/tools'
        if mode == 'train':
            self.file_folder = os.path.join(self.root_folder, 'train')
            self.class_lists = [
                'train_age_group_0.txt',
                'train_age_group_1.txt',
                'train_age_group_2.txt',
                'train_age_group_3.txt',
                'train_age_group_4.txt'
            ]
            self.pointer = [0] * 5
        else:
            self.file_folder = os.path.join(self.root_folder, 'test')
            self.class_lists = [
                'test_age_group_0.txt',
                'test_age_group_1.txt',
                'test_age_group_2.txt',
                'test_age_group_3.txt',
                'test_age_group_4.txt'
            ]
            self.pointer = [0] * 5

        self.train_label_pair = os.path.join(self.root_folder, 'train_label_pair.txt')
        self.true_labels = []
        self.false_labels = []
        self.images = []       # List of lists for each class
        self.labels = []       # Not used in current functions but kept for completeness
        self.data_size = []    # Number of images per class
        self.n_classes = classes
        self.shuffle = shuffle
        self.scale_size = scale_size
        self.label_pair_index = 0

        self.mean = np.array([104., 117., 124.])
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.z_dim = z_dim
        self.img_size = self.height

        # Uncomment and call if you want to read file lists from disk:
        # self.read_class_list(self.class_lists)
        if self.shuffle:
            self.shuffle_data(shuffle_all=True)

        self.get_age_labels()
        self.label_features_128, _ = self.pre_generate_labels(batch_size, 128, 128)
        self.label_features_64, self.one_hot_labels = self.pre_generate_labels(batch_size, 64, 64)

    def __iter__(self):
        return self

    def get_age_labels(self):
        """Generate age labels for each class as arrays of shape (batch_size,)."""
        self.age_label = [
            np.zeros(self.batch_size, np.int32),
            np.ones(self.batch_size, np.int32),
            np.ones(self.batch_size, np.int32) * 2,
            np.ones(self.batch_size, np.int32) * 3,
            np.ones(self.batch_size, np.int32) * 4
        ]

    def pre_generate_labels(self, batch_size, height, width):
        """
        Pre-generate label feature maps and one-hot labels for each class.
        
        Returns:
            batch_label_features (list): List of arrays (batch_size, height, width, n_classes).
            batch_one_hot_labels (list): List of arrays (batch_size, n_classes).
        """
        features = []
        one_hot_labels = []
        full_ones = np.ones((height, width))
        for i in range(self.n_classes):
            temp_feat = np.zeros((height, width, self.n_classes))
            temp_feat[:, :, i] = full_ones
            features.append(temp_feat)

            temp_label = np.zeros((1, self.n_classes))
            temp_label[0, i] = 1
            one_hot_labels.append(temp_label)

        batch_label_features = []
        batch_one_hot_labels = []
        for i in range(self.n_classes):
            temp_label_features = np.zeros((batch_size, height, width, self.n_classes))
            temp_one_hot = np.zeros((batch_size, self.n_classes))
            for j in range(batch_size):
                temp_label_features[j] = features[i]
                temp_one_hot[j] = one_hot_labels[i]
            batch_label_features.append(temp_label_features)
            batch_one_hot_labels.append(temp_one_hot)
        return batch_label_features, batch_one_hot_labels

    def read_class_list(self, class_lists):
        """
        Read image paths and labels from class list files.
        """
        for cl in class_lists:
            file_path = os.path.join(self.file_folder, cl)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            images_list = []
            labels_list = []
            for line in lines:
                items = line.split()
                images_list.append(items[0])
                labels_list.append(int(class_lists.index(cl)))
            self.images.append(images_list)
            self.labels.append(labels_list)
            self.data_size.append(len(labels_list))
        with open(self.train_label_pair, 'r') as f:
            lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            items = line.split()
            self.true_labels.append(int(items[0]))
            self.false_labels.append(int(items[1]))

    def next_batch(self):
        """
        Get the next batch of target images and corresponding labels.
        
        Returns:
            images (ndarray): Batch of processed images.
            batch_z (ndarray): Random noise batch.
            one_hot_labels: One-hot labels for the chosen class.
            label_features_64: Label feature map (64x64) for the chosen class.
            error_label_feature: Label feature map for a randomly chosen wrong label.
            index: Index of the chosen class.
        """
        index = random.randint(0, 4)
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        images = np.ndarray([self.batch_size, self.scale_size[0], self.scale_size[1], 3], dtype=np.float32)
        for i, p in enumerate(paths):
            images[i] = process_target_img(self.root_folder, p, self.scale_size[0])
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index] - self.batch_size):
            self.reset_pointer(index)
        label_list = [0, 1, 2, 3, 4]
        label_list.remove(index)
        random.shuffle(label_list)
        error_label = label_list[0]
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        return images, batch_z, self.one_hot_labels[index], self.label_features_64[index], \
               self.label_features_64[error_label], index

    def shuffle_data(self, index=None, shuffle_all=False):
        """Shuffle images in one or all classes."""
        if shuffle_all:
            for i in range(len(self.images)):
                random.shuffle(self.images[i])
        elif index is not None:
            random.shuffle(self.images[index])

    def reset_pointer(self, index):
        """Reset pointer for a given class and shuffle if enabled."""
        self.pointer[index] = 0
        if self.shuffle:
            self.shuffle_data(index)

    def process_source_img(self, img_path, image_size, mean, scale):
        """Process source image: read, resize, and normalize."""
        img = cv2.imread(self.root_folder + img_path)
        img = img[:, :, [2, 1, 0]]
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype(np.float32)
        return (img - mean) * scale

    def next_source_imgs(self, index, image_size, batch_size, mean=np.array([104., 117., 124.]), scale=1.):
        """
        Get the next batch of source images from a specific class.
        
        Returns:
            images: Batch of processed source images.
            paths: The file paths corresponding to these images.
        """
        paths = self.images[index][self.pointer[index]:self.pointer[index] + batch_size]
        images = np.ndarray([batch_size, image_size, image_size, 3], dtype=np.float32)
        for i, p in enumerate(paths):
            images[i] = self.process_source_img(p, image_size, mean, scale)
        self.pointer[index] += batch_size
        if self.pointer[index] >= (self.data_size[index] - batch_size):
            self.reset_pointer(index)
        return images, paths

    def load_imgs(self, data_dir, img_size=128):
        """
        Load and process a single target image.
        """
        img = cv2.imread(data_dir)
        img = img[:, :, [2, 1, 0]]
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32)
        img -= self.mean
        return img

    def load_train_imgs(self, data_dir, img_size=128):
        """
        Load all training images from a given directory.
        
        Returns:
            Numpy array of processed images.
        """
        paths = os.listdir(data_dir)
        imgs = []
        for p in paths:
            full_path = os.path.join(data_dir, p)
            img = cv2.imread(full_path)
            if img is not None:
                img = img[:, :, [2, 1, 0]]
                img = cv2.resize(img, (img_size, img_size))
                img = img.astype(np.float32)
                img -= self.mean
                imgs.append(img)
        return np.asarray(imgs)

    def save_batch(self, batch_imgs, img_names, folder, index=None, if_target=True):
        """
        Save a batch of images to disk.
        
        Args:
            batch_imgs (ndarray): Batch of images.
            img_names (list): Corresponding image names.
            folder (str): Destination folder.
            index (int, optional): Optional index to append to file names.
            if_target (bool): Flag to process target images.
        """
        assert batch_imgs.shape[0] == len(img_names), 'Number of images must match number of names'
        shape = batch_imgs.shape[1:]
        for i in range(batch_imgs.shape[0]):
            img = batch_imgs[i].reshape(shape)
            if if_target:
                im = np.uint8((img + 1.) * 127.5)
            else:
                im = np.uint8(img + self.mean)
            if im.shape[2] == 1:
                im = Image.fromarray(im.reshape([im.shape[0], im.shape[1]]), 'L')
            else:
                im = Image.fromarray(im)
            filename = os.path.join(folder, img_names[i] + ('_' + str(index) if index is not None else '') + '.jpg')
            im.save(filename)

# Utility functions outside the class

def process_target_img(root_folder, img_path, img_size):
    """
    Read and process a target image: BGR to RGB conversion, resizing, and normalization.
    """
    img = cv2.imread(root_folder + img_path)
    img = img[:, :, [2, 1, 0]]
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)
    return img / 127.5 - 1.

def process_target_img2(root_folder, img_path, img_size):
    """
    Read and process a target image for GAN use (normalized to [0,1]).
    """
    img = cv2.imread(root_folder + img_path)
    img = img[:, :, [2, 1, 0]]
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)
    return img / 255.

def process_source_img(root_folder, img_path, mean):
    """
    Read and process a source image: resize to 227x227 and subtract mean.
    """
    img = cv2.imread(root_folder + img_path)
    img = img[:, :, [2, 1, 0]]
    img = cv2.resize(img, (227, 227))
    img = img.astype(np.float32)
    return img - mean

# Uncomment the following block to run performance tests independently.
# if __name__ == '__main__':
#     generator = ImageDataGenerator(32, 128, 128, 256, shuffle=True,
#                      scale_size=(227, 227), classes=5, mode='train')
#     import time
#     time1 = time.time()
#     generator.next_batch()
#     print("next_batch time:", time.time() - time1)
#     time1 = time.time()
#     generator.mp_next_batch()
#     print("mp_next_batch time:", time.time() - time1)

# -*- coding: utf-8 -*-
"""example_project/datasets.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.02.2020

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Datasets file of example project.
"""
import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from os import path, getcwd
import glob
from PIL import Image
from torchvision import transforms


def cut_borders(image_array: np.ndarray, border_x: int, border_y: int, target_1D: bool = True):
    # Checks if the image_array is a numpy-array
    if not isinstance(image_array, np.ndarray) or len(image_array.shape) != 2:
        raise NotImplementedError("Input image-array is invalid")
    # declares some integer for our borders
    x = [0,0]
    y = [0,0]
    try:
        # Tries to parse the border-values to int
        x[0] = int(border_x[0])
        x[1] = int(border_x[1])
        y[0] = int(border_y[0])
        y[1] = int(border_y[1])
        # Check if the borders are bigger then 1
        if x[0]<1 or x[1]<1 or y[0]<1 or y[1]<1:
            raise ValueError("Border-values are too small.")
        # Check if the remaining picture is larger then
        if image_array.shape[1]-y[0]-y[1] < 16 or image_array.shape[0]-x[0]-x[1] < 16:
            raise ValueError("The remaining picture is to small, try to reduce the border.")
    except ValueError as e:
        raise ValueError(e)

    # Creates an empty known_array with zeros
    known_array = np.zeros_like(image_array)
    known_array[border_x[0]:-border_x[1], border_y[0]:-border_y[1]] = 1     # fills the array w.r.t.t. given borders
    if target_1D:
        target_array = image_array[~np.array(known_array, dtype=bool)]      # get all pixel value which will be removed
    else:
        target_array = image_array.copy()
    image_array[~np.array(known_array, dtype=bool)] = 0                     # set all pixels in the border to 0

    return image_array, known_array, target_array


def rgb2gray(rgb_array: np.ndarray, r=0.2989, g=0.5870, b=0.1140):
    """Convert numpy array with 3 color channels of shape (..., 3) to grayscale"""
    grayscale_array = (rgb_array[..., 0] * r +
                       rgb_array[..., 1] * g +
                       rgb_array[..., 2] * b)
    grayscale_array = np.round(grayscale_array)
    grayscale_array = np.asarray(grayscale_array, dtype=np.uint8)
    return grayscale_array


def custom_stacking(batch_as_list: list):

    inputs = torch.empty(size=(len(batch_as_list), 2, 90, 90), dtype=torch.float32)
    targets = torch.empty(size=(len(batch_as_list), 1, 90, 90), dtype=torch.float32)

    for i, sample in enumerate(batch_as_list):
        inputs[i, :, :, :] = torch.from_numpy(np.stack(sample[0:1], axis=0))
        targets[i, 0, :, :] = torch.from_numpy(sample[2])
    return inputs, targets


# Load data from directories
class ImageData(Dataset):
    def __init__(self, data_folder):
        p = path.join(getcwd(), data_folder, "**", "*.jpg")
        self.file_paths = glob.glob(p, recursive=True)
        self.file_paths.sort()
        self.resize = transforms.Resize((90, 90))

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx])
        img.load()
        img = self.resize(img)
        img_data = np.asarray(img, dtype="int32")
        image_array, known_array, target_array = cut_borders(img_data,
             (random.randint(5, 15), random.randint(5, 15)),
             (random.randint(5, 15), random.randint(5, 15)),
             target_1D=False
             )
        return image_array, known_array, target_array, idx


# Load data from pickle
class ImageDataTest(Dataset):
    def __init__(self, data_path):
        # Create input-batch out of data
        self.data = pickle.load(open(data_path, "rb"))

    def __len__(self):
        return len(self.data["input_arrays"])

    def __getitem__(self, idx):
        return self.data["input_arrays"][idx], self.data["known_arrays"][idx], np.empty([90, 90]), idx


class ImageNormalizer:
    """
    Class to compute the std and mean of given images
    """
    def __init__(self, data_loader):
        """
        init function of the class: stores all relative file paths of given image files
        :param input_dir: the input directory for the images to analyze
        """
        # class attributes
        self.data_loader = data_loader
        self.mean = None
        self.std = None

    def analyze_images(self):
        """
        calculate the mean and the std over all images from the class attribute file_paths
        :return: returns the mean and the std of all images in the format (mean, std)
        """
        # 2 array to store the values for every image
        mean_array = np.ndarray([len(self.data_loader.sampler)], dtype=np.float64)
        std_array = np.ndarray([len(self.data_loader.sampler)], dtype=np.float64)
        # loop over every image
        for i, batch in enumerate(self.data_loader):
            for j, picture in enumerate(batch[0]):
                mean_array[i*self.data_loader.batch_size+j] = picture[0].numpy().mean(dtype=np.float64)
                std_array[i*self.data_loader.batch_size+j] = picture[0].numpy().std(dtype=np.float64)

        # calculate the average of our 2 values
        self.mean = mean_array.mean(dtype=np.float64)
        self.std = std_array.mean(dtype=np.float64)
        return self.mean, self.std

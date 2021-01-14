# The following code is modified from https://github.com/CBICA/BrainMaGe which has the following license:

# Copyright 2020 Center for Biomedical Image Computing and Analytics, University of Pennsylvania
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# This is a 3-clause BSD license as defined in https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pandas as pd
import os
import random

import SimpleITK as sitk
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

from batchgenerators.augmentations.spatial_transformations import augment_rot90, augment_mirroring
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise

from fets.data import get_appropriate_file_paths_from_subject_dir


def completely_replace_entries(array, old_to_new):
    new_array = array.copy()
    sanity_check = np.zeros_like(array).astype(np.bool)
    for old, new in old_to_new:
        mask = array==old
        new_array[mask] = new
        sanity_check[mask]= True
    if not np.all(sanity_check):
        raise RuntimeError("Overwrite of array was incomplete.")
    return new_array


def replace_old_labels_with_new(array, class_label_map):
    # takes an old label mask and replaces with new labels
    return completely_replace_entries(array=array, old_to_new=class_label_map.items())
    

def one_hot(array, class_label_map):
    # converts newly labeled array to one-hot, skipping dimensions not represented in new labels
    new_labels = np.sort(np.unique(list(class_label_map.values())))
    new_labels_to_indices = {label: idx for idx, label in enumerate(new_labels)}

    new_label_array = replace_old_labels_with_new(array=array, class_label_map=class_label_map)
    idx_array = completely_replace_entries(array=new_label_array, old_to_new=new_labels_to_indices.items())

    # now one-hot encode using indices
    one_hot_array = np.eye(len(new_labels))[idx_array]

    # move the new (one-hot) axis to the 0th position
    axes = [-1]
    axes.extend(range(len(one_hot_array.shape))[:-1])
    one_hot_array = one_hot_array.transpose(axes)
    
    return one_hot_array


def new_labels_from_float_output(array, class_label_map, binary_classification):
    # infers class from float output (by finding new_class that is closest to float in the case (binary_classification) and
    # by taking argmax across output label dimensions and converting back to new labels otherwise)
    new_labels = np.sort(np.unique(list(class_label_map.values())))
    if binary_classification:
        # here the output has a single dim channel along 0th axis
        array = np.squeeze(array, axis=0)
        if len(new_labels) != 2:
            raise ValueError("Provided class label map does not match binary classification designation.")
        first_label, second_label = new_labels
        dist_to_first_label = np.absolute(array - first_label * np.ones_like(array))
        dist_to_second_label = np.absolute(array - second_label * np.ones_like(array))
        first_label_mask = dist_to_first_label <= dist_to_second_label
        
        # now write out the appropriate outputs
        output = array.copy()
        output[first_label_mask] = first_label
        output[~first_label_mask] = second_label
    else:
        # here the output has a multi dim channel along 0th axis 
        # (dimensions correspond to new_labels according to logic in one_hot)
        idx_array = np.argmax(array, axis=0)
        output = completely_replace_entries(array=idx_array, old_to_new=enumerate(new_labels))

    return output


"""
# TEST one_hot (though the new_labels.. test does not match the usual binary classification use case, this is ok for this unit test)
assert np.all( one_hot(np.array([1, 3]), {1:1, 3:3}) == np.array([[1, 0], [0, 1]]) )
assert np.all( new_labels_from_float_output(np.array([[1, 0], [0, 1]]), {1:1, 3:3}, False) == np.array([1, 3]) )
assert np.all( one_hot(np.array([[1, 3], [1, 1]]), {1:1, 3:3}) == np.array([[[1, 0], [1, 1]], [[0, 1], [0, 0]]]) )
assert np.all( new_labels_from_float_output(np.array([[[1, 0], [1, 1]], [[0, 1], [0, 0]]]), {1:1, 3:3}, False) == np.array([[[1, 3], [1, 1]]]) )
"""

def check_for_file_or_gzip_file(path, extensions=['.gz']):
    return find_file_or_with_extension(path, extensions) is not None


def find_file_or_with_extension(path, extensions=['.gz']):
    if os.path.exists(path):
        return path

    for ext in extensions:
        if os.path.exists(path + ext):
            return path + ext
    return None


class TumorSegmentationDataset(Dataset):
    def __init__(self, dir_paths, feature_modes, label_tags, use_case, psize, class_label_map, binary_classification, divisibility_factor=1):
        # use_case can be "training", "validation", or "inference"
        # psize determines patch size (applies only to training and validation)
        self.dir_paths = dir_paths
        self.feature_modes = feature_modes
        self.label_tags = label_tags
        self.use_case = use_case
        self.psize = psize
        self.class_label_map = class_label_map
        self.binary_classification = binary_classification
        self.divisibility_factor = divisibility_factor

    def __len__(self):
        return len(self.dir_paths)

    def transform(self,img ,gt, img_dim):
        if random.random() < 1.12:
            img, gt = augment_rot90(img, gt)
            img, gt = img.copy(), gt.copy()         
        if random.random() < 1.12:
            img, gt = augment_mirroring(img, gt)
            img, gt = img.copy(), gt.copy()
        if random.random() < 1.12:
            img = augment_gaussian_noise(img, noise_variance=(0, 0.1))
            img = img.copy()

        return img,gt

    def zero_pad(self, array):
        # zero pads in order to obtain a new array which is properly divisible in all dimensions except first
        current_shape = array.shape
        new_shape = list(current_shape)
        for idx in range(1,len(current_shape)):
            remainder = new_shape[idx] % self.divisibility_factor
            if remainder != 0: 
                new_shape[idx] += self.divisibility_factor - remainder
        zero_padded_array = np.zeros(shape=new_shape)
        slices = [slice(0,dim) for dim in current_shape]
        zero_padded_array[tuple(slices)] = array
        return zero_padded_array       
    
    @staticmethod    
    def random_slices(array, psize):
        shape = array.shape
        slices = []
        for axis, length in enumerate(psize):
            if shape[axis] > length:
                shift = random.randint(0,shape[axis] - length)
                slices.append(slice(shift, shift + length))
        return slices

    @staticmethod    
    def crop(array, slices):
        return array[tuple(slices)]

    @staticmethod
    def normalize_by_channel(array):
        # normalize to mean zero and std one per dimension along axis 0
        stack = []
        for array_slice in array:
            stack.append((array_slice - np.mean(array_slice)) / np.std(array_slice))
        return (np.stack(stack, axis=0))

    def __getitem__(self, index):
        dir_path = self.dir_paths[index]
        fname = os.path.basename(dir_path) # filename matches last dirname
        feature_stack =  []
        
        # FIXME: There is more than one place the list below is defined
        # changing the order of some definition instances (as below) effects order of channels
        # the model sees !!!
        # Move to one location and ensure sync with feature_modes from the flplan
        brats_modalities = ['T1', 'T2', 'FLAIR', 'T1CE']
        allFiles = get_appropriate_file_paths_from_subject_dir(dir_path)
        for mode in brats_modalities:
            mode_image = sitk.ReadImage(allFiles[mode])
            mode_array = sitk.GetArrayFromImage(mode_image)

            feature_stack.append(mode_array) 
        feature_array = np.stack(feature_stack)

        if self.use_case in ["training", "validation"]:
            # get label array
            label_image = None
            for label_tag in self.label_tags:
                fpath = find_file_or_with_extension(os.path.join(dir_path, fname + label_tag))
                if fpath is not None:
                    label_image = sitk.ReadImage(fpath)
                    break
            if label_image == None:
                raise RuntimeError("Data sample directory (used for train or val) missing any label with provided tags.")
            label_array = sitk.GetArrayFromImage(label_image)

            # random crop the features and labels
            slices = self.random_slices(label_array, psize=self.psize)
            label_array = self.crop(label_array, slices=slices)
            # for the feature array, we skip the first axis as it enumerates the modalities
            feature_array = self.crop(feature_array, slices = [slice(None)] + slices)

            # random crop the features and labels
            slices = self.random_slices(label_array, psize=self.psize)
            label_array = self.crop(label_array, slices=slices)
            # for the feature array, we skip the first axis as it enumerates the modalities
            feature_array = self.crop(feature_array, slices = [slice(None)] + slices)

            if self.binary_classification:
                label_array = replace_old_labels_with_new(array=label_array, class_label_map=self.class_label_map)
                label_array = np.expand_dims(label_array, axis=0)
            else:
                label_array = one_hot(array=label_array, class_label_map=self.class_label_map)
            if self.use_case == "training":
                feature_array, label_array = self.transform(img=feature_array, gt=label_array, img_dim=len(self.feature_modes))
            # normalize features
            sample = {'features': self.normalize_by_channel(feature_array), 'gt' : label_array}
        elif self.use_case == "inference":
            original_input_shape = list(feature_array.shape)
            feature_array = self.zero_pad(feature_array)
            # normalize features
            sample = {'features': self.normalize_by_channel(feature_array), 'metadata':  {"dir_path": dir_path, 
                                                                                          "original_x_dim": original_input_shape[1], 
                                                                                          "original_y_dim": original_input_shape[2], 
                                                                                          "original_z_dim": original_input_shape[3]}}
        else:
            raise ValueError("Value of TumorSegmentationDataset self.use_case is unexpected.")
        return sample


def augment_gamma(image):
    gamma = np.random.uniform(1,2)
    min_image = image.min()
    range_image = image.max() - min_image
    image = np.power(((image - min_image)/float(range_image + 1e-7)) , gamma)*range_image + min_image
    return image










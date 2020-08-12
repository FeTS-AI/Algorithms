# The following code is modified from https://github.com/CBICA/BrainMaGe which has the following lisence:

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


import scipy
import SimpleITK as sitk
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable


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
        # TODO: Enable these trainsformations (by-passed for now)
        return img,gt
        if random.random()<0.12:
            img, gt = augment_rot90(img, gt)
            img, gt = img.copy(), gt.copy()         
        if random.random()<0.12:
            img, gt = augment_mirroring(img, gt)
            img, gt = img.copy(), gt.copy()
        if random.random()<0.12:
            img = scipy.ndimage.rotate(img,45,axes=(2,1,0),reshape=False,mode='constant')
            gt = scipy.ndimage.rotate(gt,45,axes=(2,1,0),reshape=False,order=0) 
            img, gt = img.copy(), gt.copy() 
        if random.random()<0.12:
            img, gt = np.flipud(img).copy(),np.flipud(gt).copy()
        if random.random() < 0.12:
            img, gt = np.fliplr(img).copy(), np.fliplr(gt).copy()
        if random.random() < 0.12:
            for n in range(img_dim):
                img[n] = gaussian(img[n],True,0,0.1)   

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
    def rcrop(array,psize, axis_offset):
        # axis_offset is used for feature arrays (it is 1 then)
        shape = array.shape
        for axis, length in enumerate(psize):
            if shape[axis_offset + axis] > length:
                shift = random.randint(0,shape[axis_offset+axis] - length)
                slices = [slice(None) for _ in shape]
                slices[axis_offset+axis] = slice(shift, shift + length)
                array = array[tuple(slices)]
        return array

    def __getitem__(self, index):
        dir_path = self.dir_paths[index]
        fname = os.path.basename(dir_path) # filename matches last dirname
        feature_stack =  []
        for mode in self.feature_modes:
            fpath = find_file_or_with_extension(os.path.join(dir_path, fname + mode))
            if fpath is None:
                raise RuntimeError("Data sample directory missing a required image mode.")
            mode_image = sitk.ReadImage(fpath)
            mode_array = sitk.GetArrayFromImage(mode_image)

            # normalize the features for this mode
            mode_array = (mode_array - np.mean(mode_array)) / np.std(mode_array)

            feature_stack.append(mode_array) 
        feature_array = np.stack(feature_stack)

        if self.use_case in ["training", "validation"]:
            feature_array = self.rcrop(array=feature_array, psize=self.psize, axis_offset=1) 
            label_image = None
            for label_tag in self.label_tags:
                fpath = find_file_or_with_extension(os.path.join(dir_path, fname + label_tag))
                if fpath is not None:
                    label_image = sitk.ReadImage(fpath)
                    break
            if label_image == None:
                raise RuntimeError("Data sample directory (used for train or val) missing any label with provided tags.")
            label_array = sitk.GetArrayFromImage(label_image)   
            label_array = self.rcrop(array=label_array, psize=self.psize, axis_offset=0)
            if self.binary_classification:
                label_array = replace_old_labels_with_new(array=label_array, class_label_map=self.class_label_map)
            else:
                label_array = one_hot(array=label_array, class_label_map=self.class_label_map)
            if self.use_case == "training":
                feature_array, label_array = self.transform(img=feature_array, gt=label_array, img_dim=len(self.feature_modes))
            label_array = np.expand_dims(label_array, axis=0)
            sample = {'features': feature_array, 'gt' : label_array}
        elif self.use_case == "inference":
            original_input_shape = list(feature_array.shape)
            feature_array = self.zero_pad(feature_array)
            sample = {'features': feature_array, 'metadata':  {"dir_path": dir_path, 
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

def normalize(image_array):
    temp = image_array > 0
    temp_image_array = image_array[temp]    
    mu = np.mean(temp_image_array)
    sig = np.std(temp_image_array)
    image_array[temp] = (image_array[temp] - mu)/sig
    return image_array

def gaussian(img, is_training, mean, stddev):
    l,b,h =  img.shape
    noise = np.random.normal(mean, stddev, (l,b,h))
    noise = noise.reshape(l,b,h)
    img = img + noise 
    return img  



"""
Below is modified from a Creation by @author: siddhesh on Sat Jun 29 20:20:11 2019

"""

def augment_rot90(sample_data, sample_seg, num_rot=(1, 2, 3), axes=(0, 1, 2)):
    """
    :param sample_data:
    :param sample_seg:
    :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
    :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
    :return:
    """
    num_rot = np.random.choice(num_rot)
    axes = np.random.choice(axes, size=2, replace=False)
    axes.sort()
    axes = [i + 1 for i in axes]
    sample_data = np.rot90(sample_data, num_rot, axes)
    if sample_seg is not None:
        sample_seg = np.rot90(sample_seg, num_rot, axes)
    return sample_data, sample_seg


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg






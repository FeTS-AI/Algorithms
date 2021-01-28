

# TODO:insert header pointing to GANDLF repo inclusion
# TODO: Should the validation patch sampler be different from the training one?

import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
import numpy as np
import torch

from torch.utils.data import DataLoader

# put GANDLF in as a submodule staat pip install
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.parseConfig import parseConfig

from fets.data.gandlf_utils import get_dataframe_and_headers

class GANDLFData(object):

    def __init__(self, 
                 data_path, 
                 divisibility_factor,
                 **kwargs):

        self.data_path = data_path
        self.divisibility_factor = divisibility_factor

        print("\n\n###########################################################")
        print("THE PARAMS BELOW DO NOT APPLY AND SHOULD BE DISREGARDED ...")
        print("###########################################################")
        
        parameters = parseConfig(data_path['model_params_filepath'])
        
        print("###########################################################")
        print("THE PARAMS ABOVE DO NOT APPLY AND SHOULD BE DISREGARDED ...")
        print("###########################################################\n\n")

        # PATCH SAMPLING ONLY APPLIES TO THE TRAIN LOADER
        self.patch_sampler = parameters['patch_sampler']
        self.psize = parameters['psize']
        # max number of patches in the patch queue
        self.q_max_length = parameters['q_max_length']
        # number of patches to draw from a given brain volume for the queue 
        # (effects the length of the train loader)
        self.q_samples_per_volume = parameters['q_samples_per_volume']
        self.q_num_workers = parameters['q_num_workers']
        self.q_verbose = parameters['q_verbose']
        # sanity check this patch size will work with the model
        for _dim in self.psize:
            if _dim % divisibility_factor != 0:
               raise ValueError('All dimensions of the patch must be divisible by the model divisibility factor.')
        
        # set attributes that come from parameters
        self.class_list = parameters['model']['class_list']
        self.n_classes = len(self.class_list)
        # There is an assumption of batch size of 1
        self.batch_size = 1
        
        # augmentations apply only for the trianing loader
        self.train_augmentations = parameters['data_augmentation']

        self.preprocessing = parameters['data_preprocessing']

        # The data_path key, 'model_params_filepath' is required (used above), ['train', 'val', 'infernce', 'penalty'] are optional.
        # Any optionals that are not present result in associated empty loaders.
        for data_category in ['train', 'val', 'inference', 'penalty']:
            if data_category not in data_path:
                data_path[data_category] = None
        
        self.train_loader, self.penalty_loader = self.get_loader('train')
        self.val_loader, _ = self.get_loader('val')
        self.inference_loader, _ = self.get_loader('inference')
        
        self.training_data_size = len(self.train_loader)
        self.validation_data_size = len(self.val_loader)

    def get_loader(self, data_category):
        # get the data if the path is provided

        if self.data_path[data_category] is None:
            loader = []
            companion_loader = []
        else:
            if data_category == 'train':
                train = True
                augmentations = self.train_augmentations
            elif data_category == 'val':
                train = False
                augmentations = None
            elif data_category == 'inference':
                train = False
                augmentations = None
            else:
                raise ValueError('data_category needs to be one of train, val, inference, or penalty')

            DataFromPickle, headers = get_dataframe_and_headers(file_data_full=self.data_path[data_category])
            DataForTorch = ImagesFromDataFrame(dataframe=DataFromPickle, 
                                               psize=self.psize, 
                                               headers=headers, 
                                               q_max_length=self.q_max_length, 
                                               q_samples_per_volume=self.q_samples_per_volume,
                                               q_num_workers=self.q_num_workers, 
                                               q_verbose=self.q_verbose, 
                                               sampler=self.patch_sampler, 
                                               train=train, 
                                               augmentations=augmentations, 
                                               preprocessing=self.preprocessing)
            loader = DataLoader(DataForTorch, batch_size=self.batch_size)
            
            companion_loader = None
            if data_category == 'train':
                # here pinning the penalty loader to the training dataframe (the only use of companion_loader)
                # using full brain patches, and no augmentations
                CompDataForTorch = ImagesFromDataFrame(dataframe=DataFromPickle, 
                                                       psize=[240, 240, 155], 
                                                       headers=headers, 
                                                       q_max_length=self.q_max_length, 
                                                       q_samples_per_volume=self.q_samples_per_volume,
                                                       q_num_workers=self.q_num_workers, 
                                                       q_verbose=self.q_verbose, 
                                                       sampler='uniform', 
                                                       train=False, 
                                                       augmentations=None, 
                                                       preprocessing=self.preprocessing)
                companion_loader = DataLoader(CompDataForTorch, batch_size=self.batch_size)

        return loader, companion_loader

    def zero_pad(self, array, axes_to_skip=[0,1]):
        # zero pads in order to obtain a new array which is properly divisible in all appropriate dimensions
        current_shape = array.shape
        current_shape_list = list(current_shape)
        new_shape = []
        for idx, dim in enumerate(current_shape_list):
            if idx in axes_to_skip:
                new_shape.append(dim)
            else:
                remainder = dim % self.divisibility_factor
                new_shape.append(dim + self.divisibility_factor - remainder)
        zero_padded_array = torch.zeros(new_shape)
        slices = [slice(0,dim) for dim in current_shape]
        zero_padded_array[tuple(slices)] = array
        return zero_padded_array 

    def get_train_loader(self):
        return self.train_loader
    
    def get_val_loader(self):
        return self.val_loader

    def get_inference_loader(self):
        return self.inference_loader

    def get_penalty_loader(self):
        return self.penalty_loader

    def get_feature_shape(self):
        return tuple(self.psize)

    def get_training_data_size(self):
        return self.training_data_size
    
    def get_validation_data_size(self):
        return self.validation_data_size
    



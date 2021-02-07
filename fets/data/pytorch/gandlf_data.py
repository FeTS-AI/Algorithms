

# TODO:insert header pointing to GANDLF repo inclusion
# TODO: Should the validation patch sampler be different from the training one?

import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader

# put GANDLF in as a submodule staat pip install
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.parseConfig import parseConfig

from fets.data.gandlf_utils import get_dataframe_and_headers
from fets.data import get_appropriate_file_paths_from_subject_dir



    


class GANDLFData(object):

    def __init__(self, 
                 data_path, 
                 divisibility_factor,
                 patch_sampler, 
                 psize, 
                 q_max_length, 
                 q_samples_per_volume, 
                 q_num_workers, 
                 q_verbose, 
                 class_list, 
                 data_augmentation, 
                 data_preprocessing,
                 excluded_subdirs = ['log', 'logs'],
                 percent_train = 0.8,
                 in_memory=False,
                 data_usage='train-val', 
                 shuffle_before_train_val_split=True,
                 **kwargs):

        # some hard-coded atributes
        # feature stack order (determines order of feature stack modes)
        # depenency here with mode naming convention used in get_appropriate_file_paths_from_subject_dir
        self.feature_modes = ['T1', 'T2', 'FLAIR', 'T1CE']
        # used as headers for dataframe used to create data loader (when csv's are not provided)
        # dependency (other methods expect the first header to be subject name and subsequent ones being self.feature_modes)
        self.default_dataframe_headers = ['SubjectID'] + self.feature_modes

        self.divisibility_factor = divisibility_factor
        self.in_memory = in_memory
        
        # PATCH SAMPLING ONLY APPLIES TO THE TRAIN LOADER
        self.patch_sampler = patch_sampler
        self.psize = psize
        # max number of patches in the patch queue
        self.q_max_length = q_max_length
        # number of patches to draw from a given brain volume for the queue 
        # (effects the length of the train loader)
        self.q_samples_per_volume = q_samples_per_volume
        self.q_num_workers = q_num_workers
        self.q_verbose = q_verbose
        # sanity check this patch size will work with the model
        for _dim in self.psize:
            if _dim % divisibility_factor != 0:
               raise ValueError('All dimensions of the patch must be divisible by the model divisibility factor.')
        
        self.class_list = class_list
        self.n_classes = len(self.class_list)
        # There is an assumption of batch size of 1
        self.batch_size = 1
        
        # augmentations apply only for the trianing loader
        self.train_augmentations = data_augmentation

        self.preprocessing = data_preprocessing

        # sudirectories to skip when listing patient subdirectories inside data directory
        self.excluded_subdirs = excluded_subdirs
        self.shuffle_before_train_val_split = shuffle_before_train_val_split

        self.data_files

        here create dictionary of dataframes (which can be none)
        the dataframes can come from csvs, will happen if data_name
        is a dictionary.

        

        if data_usage == 'train-val':
            if isinstance(data_path, dict):
                if ('train' not in data_path) or ('val' not in data_path):
                    raise ValueError('Must point to csv for both train and val when data_usage is train-val and using dictionary data path.')
                train_dataframe, train_headers = get_dataframe_and_headers(file_data_full=data_path['train'])
                val_dataframe,  val_headers = get_dataframe_and_headers(file_data_full=data_path['val'])
                
                # validate headers are the same
                if len(train_headers) != len(val_headers):
                    raise ValueError('Train/Val headers must align, but found different number of headers in each.')
                for idx in len(train_headers):
                    if train_headers[idx] != val_headers[idx]:
                        raise ValueError('Train/Val headers must align ({} != {} found at index {}).'.format(train_headers[idx], val_headers[idx], idx))
                headers = train_headers
            else:
                headers = self.default_dataframe_headers
                train_dataframe, val_dataframe = self.create_train_val_dataframes(pardir=datapath, headers=headers, percent_train=percent_train)
            self.headers = headers
        elif data_usage == 'inference':
        else:
            raise ValueError('data_usage needs to be either train-val or inference')

        WORKING HERE TOO
            for

delete below
        # The data_path key, 'model_params_filepath' is required (used above), ['train', 'val', 'infernce', 'penalty'] are optional.
        # Any optionals that are not present result in associated empty loaders.
        for data_category in ['train-val', 'inference', 'penalty']:
            if data_category not in data_path:
                data_path[data_category] = None
        
        self.train_loader, self.penalty_loader = self.get_loader('train')
        self.val_loader, _ = self.get_loader('val')
        self.inference_loader, _ = self.get_loader('inference')
        
        self.training_data_size = len(self.train_loader)
        self.validation_data_size = len(self.val_loader)

    def create_dataframe_from_subdirpaths(self, subdir_paths):
        columns = {header: [] for header in self.default_dataframe_headers}
        for subdir_path in subdir_paths:
            # grab second to last part of path (subdir name)
            subdir = os.path.split(os.path.split(subdir_path)[0])[1]
            feature_fpaths = get_appropriate_file_paths_from_subject_dir(subdir_path)
            # write dataframe row
            colums[self.default_dataframe_headers[0]] = subdir
            for mode in self.feature_modes:
                columns[mode].append(feature_fpaths[mode])

            # sanity check that you have writen to every column of this row
            comparison = None
            for key, column in columns.items:
                this_length = len(column)
                if comparison is not None:
                    if this_length != comparison:
                        raise RuntimeError('Filling dataframe for dataset and attempted to progress without writing entire row.')
                comparison = this_length

        return pd.DataFrame(columns)

    def create_train_val_dataframes(self, pardir, headers, percent_train):
        subdirs_list = os.listdir(pardir)
        # filter subdirectories not meant for grabbing subject data
        sudirs_list = [subdir for subdir in subdirs_list if subdir not in self.excluded_subdirs]
        total_subjects = len(subdirs_list)
        
        # create full paths to subdirs
        subdir_paths_list = [os.path.join(pardir, subdir) for sudir in subdirs_list]
        
        if self.shuffle_before_train_val_split:
            np.random.shuffle(subdir_paths_list)
        
        # compute the split

        # sanity checks
        if (percent_train <= 0.0) or (percent_train >= 1.0):
            raise ValueError('Value of percent_train must be stricly between 0 and 1.') 
        split_idx = int(percent_train * total_subjects)
        if (split_idx == 0) or (split_idx == total_subjects):
            raise ValueError('Value of percent_train {} is leading to empty val or train set due to only {} subjects found under {} '.format(percent_train, total_subjects, pardir))
        
        train_paths = sudir_paths_list[:split_idx]
        val_paths = subdir_paths_list[split_idx:] 
        print('Splitting the {} subjects in {} using percent_train of {}'.format(total_subjects, pardir, percent_train))
        print('Resulting train and val sets have counts {} and {} respectively.'.format(len(train_paths), len(val_paths)))
        
        # create the dataframes
        train_dataframe = create_dataframe_from_subdirpaths(train_paths)
        val_dataframe = create_dataframe_from_subdirpaths(val_paths)

        return train_dataframe, val_dataframe
        


WORKING HERE

        # create val dataframe


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
                                               preprocessing=self.preprocessing, 
                                               in_memory=self.in_memory)
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
    



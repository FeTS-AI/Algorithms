
# TODO:insert header pointing to GANDLF repo inclusion
# TODO: Should the validation patch sampler be different from the training one?
# FIXME: replace prints to stdout with logging.

import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
import numpy as np
import torch
import torchio
import pandas as pd
import pickle as pkl

from torch.utils.data import DataLoader

# put GANDLF in as a submodule staat pip install
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.parseConfig import parseConfig

from fets.data.gandlf_utils import get_dataframe_and_headers
from fets.data import get_appropriate_file_paths_from_subject_dir


# adapted from https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132933#132933
def crop_image_outside_zeros(array, psize):
    dimensions = len(array.shape)
    if dimensions != 4:
        raise ValueError("Array expected to be 4D but got {} dimensions.".format(dimensions)) 
    
    # collapse to single channel and get the mask of non-zero voxels
    mask = array.sum(axis=0) > 0

    # get the small and large corners

    m0 = mask.any(1).any(1)
    m1 = mask.any(0)
    m2 = m1.any(0)
    m1 = m1.any(1)
    
    small = [m0.argmax(), m1.argmax(), m2.argmax()]
    large = [m0[::-1].argmax(), m1[::-1].argmax(), m2[::-1].argmax()]
    large = [m - l for m, l in zip(mask.shape, large)]
    
    # ensure we have a full patch
    # for each axis
    for i in range(3):
        # if less than patch size, extend the small corner out
        if large[i] - small[i] < psize[i]:
            small[i] = large[i] - psize[i]

        # if bottom fell off array, extend the large corner and set small to 0
        if small[i] < 0:
            small[i] = 0
            large[i] = psize[i]

    # calculate pixel location of new bounding box corners
    small_idx_corner = [small[0], small[1], small[2]]
    large_idx_corner = [large[0], large[1], large[2]]
    # Get the contents of the bounding box from the array
    new_array = array[:,
                    small[0]:large[0],
                    small[1]:large[1],
                    small[2]:large[2]]
    
    return small_idx_corner, large_idx_corner, new_array


def load_pickle(fpath):
    if os.path.exists(fpath):
        with open(fpath, 'rb') as _file:
            return pkl.load(_file)
    else:
        return None


def dump_pickle(objct, path):
    with open(path, 'wb') as _file:
        pkl.dump(objct, _file)


def dataframe_as_string(dataframe):
    return dataframe.astype(str)


def dataframe_to_string_csv(dataframe, path):
    dataframe_as_string(dataframe).to_csv(path, index=False)    


class GANDLFData(object):

    def __init__(self, 
                 data_path,
                 class_list=[0, 1, 2, 4], 
                 patch_sampler='uniform',       
                 psize=[128, 128, 128],
                 divisibility_factor=16,
                 q_samples_per_volume=1,
                 q_verbose=False, 
                 data_augmentation=None, 
                 data_preprocessing=None,
                 split_instance_dirname='default_split_instance',
                 np_split_seed=9264097,
                 handle_missing_datafiles=False,
                 q_max_length=1,
                 q_num_workers=0,
                 excluded_subdirs = ['log', 'logs'],
                 percent_train = 0.8,
                 in_memory=False,
                 data_usage='train-val',
                 shuffle_before_train_val_split=True,
                 allow_new_data_into_previous_split = True,
                 handle_data_loss_from_previous_split= False,
                 **kwargs):

        # some hard-coded atributes
        # feature stack order (determines order of feature stack modes)
        # depenency here with mode naming convention used in get_appropriate_file_paths_from_subject_dir
        self.feature_modes = ['T1', 'T2', 'FLAIR', 'T1CE']
        self.label_tag = 'Label'

        self.data_path = data_path

        # using numerical header names
        self.numeric_header_names = {mode: idx+1 for idx, mode in enumerate(self.feature_modes)}
        self.numeric_header_names[self.label_tag] = len(self.feature_modes) + 1
        # inverse of dictionary above
        self.numeric_header_name_to_key = {value: key for key, value in self.numeric_header_names.items()}
        
        # used as headers for dataframe used to create data loader (when csv's are not provided)
        # dependency (other methods expect the first header to be subject name and subsequent ones being self.feature_modes)
        
        self.train_val_headers = {}
        self.train_val_headers['subjectIDHeader'] = 0
        self.train_val_headers['channelHeaders'] = [self.numeric_header_names[mode] for mode in self.feature_modes]
        self.train_val_headers['labelHeader'] = self.numeric_header_names[self.label_tag]
        self.train_val_headers['predictionHeaders'] = []
        
        self.inference_headers = {}
        self.inference_headers['subjectIDHeader'] = 0
        self.inference_headers['channelHeaders'] = [self.numeric_header_names[mode] for mode in self.feature_modes]
        self.inference_headers['labelHeader'] = None
        self.inference_headers['predictionHeaders'] = []
        

        self.divisibility_factor = divisibility_factor
        self.in_memory = in_memory
        
        # patch sampling only applies to the train loader (when used at all)
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
        self.split_instance_dirname = split_instance_dirname
        self.shuffle_before_train_val_split = shuffle_before_train_val_split
        self.random_generator_instance = np.random.default_rng(np_split_seed)

        self.percent_train = percent_train
        # sanity check
        if (self.percent_train <= 0.0) or (self.percent_train >= 1.0):
            raise ValueError('Value of percent_train must be stricly between 0 and 1.')

        # do we allow new data into training and validation (when some train and val was previously assigned)
        self.allow_new_data_into_previous_split = allow_new_data_into_previous_split
        # do we throw exceptions for missing disk data previously recorded as train or val samples,
        # or record information in a file and do our best to restore percent_train with new samples
        self.handle_data_loss_from_previous_split = handle_data_loss_from_previous_split

        # do we throw exceptions for data subdirectories with missing files, or just skip them
        self.handle_missing_datafiles = handle_missing_datafiles

        # hard-coded file name
        self.split_info_dirname = 'split_info'
        
        # sudirectories to skip when listing patient subdirectories inside data directory
        self.excluded_subdirs = excluded_subdirs
        # append the split info directory
        self.excluded_subdirs.append(self.split_info_dirname)
        
        if data_usage == 'train-val':
            self.setup_for_train_val()
        elif data_usage == 'inference':
            self.setup_for_inference()        
        else:
            raise ValueError('data_usage needs to be either train-val or inference')
        
        self.training_data_size = len(self.train_loader)
        self.validation_data_size = len(self.val_loader)

    def setup_for_train_val(self):
        self.inference_loader = []
            
        self.set_dataframe_headers(self.train_val_headers, list_needed=True)
        
        # FIXME: below contains some hard coded file names
        self.set_split_info_paths()
        
        past_split_exists = self.sanity_check_split_info()
        if not past_split_exists:
            self.allow_previously_unseen_data = True
        else:
            self.allow_previously_unseen_data = self.allow_new_data_into_previous_split

        on_disk = self.get_data_from_disk()
        self.num_on_disk = len(on_disk)

        past_train, past_val = self.get_past_data()

        lost_train, lost_val = self.compare_disk_to_past(on_disk, past_train, past_val)
   
        init_train = on_disk.intersection(past_train)
        self.num_train_historical_assignments = len(init_train)
        
        init_val = on_disk.intersection(past_val)
        self.num_val_historical_assignments = len(init_val)
        

        if self.allow_previously_unseen_data:
            to_split = on_disk - init_train - init_val
            self.num_fresh_to_split = len(to_split)
            
            new_train, new_val = self.split(list(to_split), 
                                            num_existing_train=len(init_train), 
                                            num_existing_val=len(init_val))
            self.num_train_fresh_assignments = len(new_train)
            self.num_val_fresh_assignments = len(new_val)

            train = list(init_train.union(set(new_train)))    
            val = list(init_val.union(set(new_val)))          
        else:
            train = list(init_train)
            val = list(init_val)

        self.check_for_undesirable_split(train=train, val=val)

        # writing out lost data and split info 
        self.record_lost_data(lost_train=list(lost_train), lost_val=list(lost_val))
        self.record_split_info(train=train, val=val)
        
        # constructing dataframes, then loaders from the dataframes
        train_dataframe, val_dataframe = self.create_train_val_dataframes(train_subdirs=train, val_subdirs=val)     
        self.set_train_and_val_loaders(train_dataframe=train_dataframe, val_dataframe=val_dataframe)

    def get_data_from_disk(self, include_labels=True, return_as_set=True):
        good_subdirs = []
   
        subdirs_to_fpaths = {}
        incomplete_subdirs = []
        for subdir in self.get_sorted_subdirs():
            subdir_path = os.path.join(self.data_path, subdir)
            fpaths = get_appropriate_file_paths_from_subject_dir(dir_path=subdir_path, 
                                                                 include_labels=include_labels,         
                                                                 handle_missing_datafiles=self.handle_missing_datafiles)
            if fpaths is None:
                if self.handle_missing_datafiles:
                    incomplete_subdirs.append(subdir_path)
                    continue
                else:
                    raise ValueError('Unexpected None return from get_appropriate_fiile_paths_from_subject_dir')
            else:
                good_subdirs.append(subdir)
                subdirs_to_fpaths[subdir] = fpaths
        if len(incomplete_subdirs) != 0:
            print('\nIgnoring subdirecories: {} as they are missing needed data files.\n'.format(incomplete_subdirs))
        if return_as_set:
            good_subdirs = set(good_subdirs)

        if hasattr(self, 'subdirs_to_fpaths'):
            raise RuntimeError('Use-cases comprehended for attribute subdirs_to_fpaths is currently only write-once.')
        self.subdirs_to_fpaths = subdirs_to_fpaths
        
        return good_subdirs
                       
    def subdirs_to_subdirpaths(self, subdirs):
        return [os.path.join(self.data_path, subdir) for subdir in subdirs]

    def subdirpaths_to_subdirs(self, subdirpaths):
        return [os.path.split(path)[1] for path in subdirpaths]

    def split(self, subdirs, num_existing_train, num_existing_val):

        if len(subdirs) == 0:
            return [], []
        else:

            # sorting to make process deterministic for a fixed seed
            subdirs = np.sort(subdirs)
            total_subjects = len(subdirs) + num_existing_train + num_existing_val
            if self.shuffle_before_train_val_split:  
                self.random_generator_instance.shuffle(subdirs)
                # cast back to a list if needed
                subdirs = subdirs.tolist()

            # we want: (new_train + exist_train)/tot ~ percent_train ; so new_train ~ tot * percent_train - existing_train
            split_idx = int(self.percent_train * total_subjects - num_existing_train)
            train_subdirs = subdirs[:split_idx]
            val_subdirs = subdirs[split_idx:]
    
            return train_subdirs, val_subdirs   
        
    def sanity_check_split_info(self):
        past_split_exists = False
        # check for some cases we will not tolerate
        if os.path.exists(self.split_instance_dirpath):
            need_all = [self.train_csv_path, self.val_csv_path, self.pickled_split_path]
            need_all_or_none = [self.pickled_lost_train_path, self.pickled_lost_val_path]
            exists_in_all = [os.path.exists(_path) for _path in need_all]
            exists_in_all_or_none = [os.path.exists(_path) for _path in need_all_or_none]
            if not np.all(exists_in_all):
                raise ValueError('At least one of {} is missing!! Carefully recover (contents were printed to stdout during last run).'.format(need_all))
            if np.any(exists_in_all_or_none) and not np.all(exists_in_all_or_none):
                raise ValueError('All or none of: {} should exists, but exactly one was found!! Carefully recover (contents were printed to stdout during last run).'.format(need_all_or_none))
            past_split_exists = True
        return past_split_exists

    def get_split_info(self):
        self.sanity_check_split_info()
        # get split from csvs and pickle (above check confirms they exist if split instance does)
        if not os.path.exists(self.split_instance_dirpath):
            subdirs_only_csv_train = []
            subdirs_only_csv_val = []
            pickled_train = []
            pickled_val = []
        else:
            csv_train_dataframe = dataframe_as_string(pd.read_csv(self.train_csv_path))
            csv_val_dataframe = dataframe_as_string(pd.read_csv(self.val_csv_path))
            subdirs_only_csv_train = list(csv_train_dataframe[str(self.train_val_headers['subjectIDHeader'])])
            subdirs_only_csv_val = list(csv_val_dataframe[str(self.train_val_headers['subjectIDHeader'])])

            pickled_train, pickled_val = load_pickle(self.pickled_split_path)
            
        # check for consistency
        if subdirs_only_csv_train != pickled_train:
            raise ValueError('Train csv and pickled split info do not match. Carefully recover (contents were printed to stdout during last run).')
        if subdirs_only_csv_val != pickled_val:
            raise ValueError('Validation csv and pickled split info do not match. Carefully recover (contents were printed to stdout during last run).')
        return sorted(subdirs_only_csv_train), sorted(subdirs_only_csv_val)

    def get_past_data(self):
        split_train, split_val = self.get_split_info()
        lost_train, lost_val = self.get_lost_data()
        
        # combine old split with lost data
        past_train = set(split_train).union(set(lost_train))
        past_val = set(split_val).union(set(lost_val))

        # sanity check
        if past_train.intersection(past_val) != set():
            raise ValueError('Somehow there is a record of training intersecting validation.')

        return past_train, past_val

    def compare_disk_to_past(self, on_disk, past_train, past_val):

        lost_train = past_train - on_disk
        lost_val = past_val - on_disk

        if (lost_train != set([])) or (lost_val != set([])):
            # sanity check
            if not os.path.exists(self.split_instance_dirpath):
                raise RuntimeError('Claiming lost data when no historical data info exists.')
            
            if lost_train != set([]):
                print('\nWARNING: Train samples: {} from split: {} now missing on disk.\n'.format(list(lost_train), self.split_instance_dirname))
                if not self.handle_data_loss_from_previous_split:
                    raise ValueError('Train samples: {} from split: {} now missing on disk and allow_data_loss is False.'.format(list(lost_train), self.split_instance_dirname))
            if lost_val != set([]):
                print('\nWARNING: Val samples: {} from split: {} now missing on disk.\n'.format(list(lost_val), self.split_instance_dirname))
                if not self.handle_data_loss_from_previous_split:
                    raise ValueError('Val samples: {} from split: {} now missing on disk and allow_data_loss is False.'.format(list(lost_val), self.split_instance_dirname))

        return lost_train, lost_val           

    def get_lost_data(self):
        self.sanity_check_split_info()
        # get lost data info from pickle (above check confirms either both lost data files or neither exists)
        lost_train = load_pickle(self.pickled_lost_train_path) or []
        lost_val = load_pickle(self.pickled_lost_val_path) or []

        return sorted(lost_train), sorted(lost_val)

    def record_split_info(self, train, val):

        train = sorted(train)
        val = sorted(val)
        print('\n Data split information:')
        print('\n{} good data subdirectories were found in {}'.format(self.num_on_disk, self.data_path))
        if self.num_train_historical_assignments + self.num_val_historical_assignments != 0:
            print('{} of these were known to be used previously, so were assigned to train/val using historical split info.'.format(self.num_train_historical_assignments + self.num_val_historical_assignments))
        if self.allow_previously_unseen_data and self.num_fresh_to_split != 0:
            print('{} of these were previosly unknown, so assigned randomly (with best effort to maintain percent train:{}.'.format(self.num_fresh_to_split, self.percent_train))
        elif not self.allow_previously_unseen_data and (self.num_on_disk > self.num_train_historical_assignments + self.num_val_historical_assignments):
            print('Not allowing previously unseen data samples into split, since old split info exists and allow_new_data_into_previous_split is False.')
        
        print('\nSubdirectories used for training: {}'.format(train))
        if self.num_train_historical_assignments != 0:
            print('{} assigned using previous split info.'.format(self.num_train_historical_assignments))
        if self.allow_previously_unseen_data and (self.num_train_fresh_assignments != 0):
            print('{} newly assigned.'.format(self.num_train_fresh_assignments))


        print('\nSubdirectories used for validation: {}'.format(val))
        if self.num_val_historical_assignments != 0:
            print('{} assigned using previous split info.'.format(self.num_val_historical_assignments))
        if self.allow_previously_unseen_data and (self.num_val_fresh_assignments != 0):
            print('{} newly assigned.\n'.format(self.num_val_fresh_assignments))

        if not os.path.exists(self.split_instance_dirpath):
            previous_split = False
        else:
            previous_split = True

        # the only case that we do not write out is when previous split exists and matches current split
        write_out = True
        if previous_split:
            prev_train, prev_val = self.get_split_info()
            if (prev_train == train) and (prev_val == val):
                write_out = False

        if write_out:
            if not os.path.exists(self.split_instance_dirpath):
                os.mkdir(self.split_instance_dirpath)
            dump_pickle((list(train), list(val)), path=self.pickled_split_path) 
            temp_train_dataframe, temp_val_dataframe = self.create_train_val_dataframes(train_subdirs=train, 
                                                                                        val_subdirs=val)
            dataframe_to_string_csv(dataframe=temp_train_dataframe, path=self.train_csv_path)
            dataframe_to_string_csv(dataframe=temp_val_dataframe, path=self.val_csv_path)
    
    def record_lost_data(self, lost_train, lost_val):
        lost_train = sorted(lost_train)
        lost_val = sorted(lost_val)

        lists_empty = (len(lost_train) + len(lost_val) == 0)  
        self.sanity_check_split_info()
        
        # sanity check above ensures either one or both of lost data files are present
        old_lost_info_present = os.path.exists(self.pickled_lost_train_path)
        
        if (not lists_empty) or old_lost_info_present:
            print('\nLost data info:')
            print('Subdirectories: {} were previously used for training but now missing in {}'.format(lost_train, self.data_path))
            print('Subdirectories: {} were previously used for validation but now missing in {}\n'.format(lost_val, self.data_path)) 

            write_out = False

            if not old_lost_info_present:
                # here the lists are not empty and we have no previously recorded lost data
                write_out = True
            else: 
                # here previous lost info is present, and so we can compare previous with current
                previous_lost_train, previous_lost_val = self.get_lost_data()
                if (previous_lost_train != lost_train) or (previous_lost_val != lost_val):
                    write_out=True

            if write_out:
                if not os.path.exists(self.split_instance_dirpath):
                    os.mkdir(self.split_instance_dirpath)
                dump_pickle(list(lost_train), path=self.pickled_lost_train_path)
                dump_pickle(list(lost_val), path=self.pickled_lost_val_path) 

    def check_for_undesirable_split(self, train, val):

        if (len(train) ==0) or (len(val) == 0):
                raise ValueError('Split (accounting for percent_train, historical train/val assignments, and complete data subdirectories under data_path) results in an empty train or val set.')

    def set_train_and_val_loaders(self, train_dataframe, val_dataframe):
        self.train_loader, self.penalty_loader = self.get_loaders(data_frame=train_dataframe, train=True, augmentations=self.train_augmentations)
        self.val_loader, _ = self.get_loaders(data_frame=val_dataframe, train=False, augmentations=None)
      
    def set_dataframe_headers(self, headers, list_needed=False):
        self.headers = headers
        if list_needed:
            self.headers_list = [self.headers['subjectIDHeader']] + self.headers['channelHeaders'] + [self.headers['labelHeader']] + self.headers['predictionHeaders']

    def set_split_info_paths(self):

        # hard coded file names
        train_csv_fname = 'train.csv'
        val_csv_fname = 'val.csv'
        pickled_split_fname = 'pickled_train_val_lists_tuple.pkl'
        lost_train_fname = 'lost_train.pkl'
        lost_val_fname = 'lost_val.pkl'

        # derived paths
        split_info_dirpath = os.path.join(self.data_path, self.split_info_dirname)
        self.split_instance_dirpath = os.path.join(split_info_dirpath, self.split_instance_dirname)
        self.train_csv_path = os.path.join(self.split_instance_dirpath, train_csv_fname)
        self.val_csv_path = os.path.join(self.split_instance_dirpath, val_csv_fname)
        self.pickled_split_path = os.path.join(self.split_instance_dirpath, pickled_split_fname)
        self.pickled_lost_train_path = os.path.join(self.split_instance_dirpath, lost_train_fname)
        self.pickled_lost_val_path = os.path.join(self.split_instance_dirpath, lost_val_fname)

        if not os.path.exists(split_info_dirpath):
            os.mkdir(split_info_dirpath)


    def create_dataframe(self, subdirs, include_labels):

        columns = {header: [] for header in self.headers_list if (header is not None)}
        for subdir in subdirs:
            fpaths = self.subdirs_to_fpaths[subdir] 
            # write dataframe row
            columns[self.headers_list[0]].append(subdir)
            for header in self.headers_list[1:]:
                if header is not None:
                    columns[header].append(fpaths[self.numeric_header_name_to_key[header]])
        return pd.DataFrame(columns)

    def get_sorted_subdirs(self):
        subdirs_list = os.listdir(self.data_path)

        # filter subdirectories not meant for grabbing subject data
        subdirs_list = np.sort([subdir for subdir in subdirs_list if subdir not in self.excluded_subdirs])
        
        return subdirs_list

    def create_train_val_dataframes(self, train_subdirs, val_subdirs):
                
        # create the dataframes
        train_dataframe = self.create_dataframe(train_subdirs, include_labels=True)
        val_dataframe = self.create_dataframe(val_subdirs, include_labels=True)

        return train_dataframe, val_dataframe

    def setup_for_inference(self):
        self.set_dataframe_headers(self.inference_headers, list_needed=True)
        inference_dataframe = self.create_inference_dataframe()
        
        self.train_loader = []
        self.penalty_loader = []
        self.val_loader = []
        self.inference_loader, _ = self.get_loaders(data_frame=inference_dataframe, train=False, augmentations=None)

    
    def create_inference_dataframe(self):
        inference_subdirs = self.get_data_from_disk(include_labels=False, return_as_set=False)
        # create the dataframes
        inference_dataframe = self.create_dataframe(inference_subdirs, include_labels=False)
    
        return inference_dataframe
        
    def get_loaders(self, data_frame, train, augmentations):
        
        data = ImagesFromDataFrame(dataframe=data_frame, 
                                   psize=self.psize, 
                                   headers=self.headers, 
                                   q_max_length=self.q_max_length, 
                                   q_samples_per_volume=self.q_samples_per_volume,
                                   q_num_workers=self.q_num_workers, 
                                   q_verbose=self.q_verbose, 
                                   sampler=self.patch_sampler, 
                                   train=train, 
                                   augmentations=augmentations, 
                                   preprocessing=self.preprocessing, 
                                   in_memory=self.in_memory)
        loader = DataLoader(data, batch_size=self.batch_size)
        
        companion_loader = None
        if train:
            # here pinning the penalty loader to the training dataframe (the only use of companion_loader)
            # using full brain patches, and no augmentations
            companion_data = ImagesFromDataFrame(dataframe=data_frame, 
                                                 psize=[240, 240, 155], 
                                                 headers=self.headers, 
                                                 q_max_length=self.q_max_length, 
                                                 q_samples_per_volume=self.q_samples_per_volume,
                                                 q_num_workers=self.q_num_workers, 
                                                 q_verbose=self.q_verbose, 
                                                 sampler='uniform', 
                                                 train=False, 
                                                 augmentations=None, 
                                                 preprocessing=self.preprocessing)
            companion_loader = DataLoader(companion_data, batch_size=self.batch_size)

        return loader, companion_loader

    def zero_pad(self, tensor, axes_to_skip=[0]):
        # zero pads in order to obtain a new array which is properly divisible in all appropriate dimensions
        # padding is done on top of highest indices across all dimensions
        current_shape = tensor.shape
        current_shape_list = list(current_shape)
        new_shape = []
        for idx, dim in enumerate(current_shape_list):
            if idx in axes_to_skip:
                new_shape.append(dim)
            else:
                remainder = dim % self.divisibility_factor
                indices_to_add = (self.divisibility_factor - remainder) % self.divisibility_factor
                new_shape.append(dim + indices_to_add)
        zero_padded_tensor = torch.zeros(new_shape)
        slices = [slice(0,dim) for dim in current_shape_list]
        zero_padded_tensor[tuple(slices)] = tensor
        return zero_padded_tensor 

    def infer_with_patches(self, model_inference_function, features):
        # This function infers using multiple patches, fusing corresponding outputs

        # model_inference_function is a list to suport recursive calls to similar function

        subject_dict = {}
        for i in range(0, features.shape[1]): # 0 is batch
            subject_dict[str(i)] = torchio.Image(tensor = features[:,i,:,:,:], type=torchio.INTENSITY)
        
        grid_sampler = torchio.inference.GridSampler(torchio.Subject(subject_dict), self.psize)
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        aggregator = torchio.inference.GridAggregator(grid_sampler)

        for patches_batch in patch_loader:
            # concatenate the different modalities into a tensor
            image = torch.cat([patches_batch[str(i)][torchio.DATA] for i in range(0, features.shape[1])], dim=1)
            locations = patches_batch[torchio.LOCATION] # get location of patch
            pred_mask = model_inference_function[0](model_inference_function=model_inference_function[1:], features=image)
            aggregator.add_batch(pred_mask, locations)
        output = aggregator.get_output_tensor() # this is the final mask
        output = output.unsqueeze(0) # increasing the number of dimension of the mask
        return output

    def infer_with_crop_and_patches(self, model_inference_function,features):
        # crops external zero-planes (tracking indices cropped), infers the cropped image with patches, then pads the output 
        # with zeros to restore the original shape

        return self.infer_with_crop(model_inference_function=[self.infer_with_patches] + model_inference_function, features=features)
        

    def infer_with_crop(self, model_inference_function, features):
        # crops external zero-planes (tracking indices cropped), infers the cropped image in one pass, then pads the output 
        # with zeros to restore the original shape

        # model_inference_function is a list to suport recursive calls to similar function

        # record original feature shape
        original_shape = list(features.shape)
        # sanity check on expected shape (assumptions used to detect physical dimensions to construct a properly shaped output)
        if len(original_shape) != 5:
            raise ValueError('Expected features shape to be of length 5.')
        if original_shape[0] != 1:
            raise ValueError('Expected batch dimension to be 1 in features.')
        if original_shape[1] != len(self.feature_modes):
            raise ValueError('Expected scaning modes to be eunumerated along axis 1 of features.')

        # crop function will expect a numpy array, and that the batch dimension is stripped
        features_np = features.numpy().copy()
        squeezed_features_np = np.squeeze(features_np, axis=0)
        small_idx_corner, large_idx_corner, cropped_features_np = crop_image_outside_zeros(array=squeezed_features_np, psize = self.psize)
        # convert back to torch tensor
        cropped_features = torch.tensor(cropped_features_np)
        

        # we will not track how many indices are added during this padding, as the associated output
        # indices will be ignored (since these are associated with input pixels of zeros, they will
        # either get replaced by output of zero or dropped when they extend above original image size)
        final_features = self.zero_pad(tensor=cropped_features)

        # put back in batch dimension (including in cropped idx info)
        final_features = final_features.unsqueeze(0)

        # perform inference
        output_of_cropped = model_inference_function[0](features=final_features, model_inference_function=model_inference_function[1:])
        # some sanity checks using our assumptions of: 
        #     5 total axes: (batches=1, num_classes, spacial_x, spacial_y, spacial_z)
        prelim_shape = output_of_cropped.shape
        if (len(prelim_shape) != 5) or (prelim_shape[0] != 1) or (prelim_shape[1] != len(self.class_list)):
            raise ValueError('Expected shape [1, num_classes, spacial_x, spacial_y, spacial_z] of cropped-feature output and found ', output_of_cropped.shape)
        
        output_shape = [1, len(self.class_list), original_shape[2], original_shape[3], original_shape[4]] 

        # prepare final output by initializing with all background (using appropriate class encoding)
        # checking against two use-cases (will need to change to accomadate others)
        output = torch.zeros(size=output_shape)
        if self.class_list == [0, 1, 2 , 4]:
            # in this case, background is encoded in the first output channel
            output[:,0,:,:,:] = 1
        elif not (set(self.class_list) == set(['4', '1||2||4', '1||4'])):
            raise ValueError('Supporting class list of {} is not present.'.format(self.class_list))
  
        # write in non-background output using the output of cropped features
        output[:, :, small_idx_corner[0]:large_idx_corner[0],small_idx_corner[1]:large_idx_corner[1], small_idx_corner[2]:large_idx_corner[2]] = \
            output_of_cropped[:,:,:large_idx_corner[0]-small_idx_corner[0],:large_idx_corner[1]-small_idx_corner[1],:large_idx_corner[2]-small_idx_corner[2]]
        
        return output

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
    



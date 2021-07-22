
# TODO:insert header pointing to GANDLF repo inclusion
# TODO: Should the validation patch sampler be different from the training one?

import os, pathlib
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
import logging
import numpy as np
import torch
import torchio
import pandas as pd
import random

from torch.utils.data import DataLoader

import SimpleITK as sitk

# put GANDLF in as a submodule staat pip install
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame

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


# used in two functions below
single_column_name = 'data_uid'


def write_list_to_single_column_csv(_list, fpath):
    dataframe = pd.DataFrame({single_column_name: _list})
    dataframe.to_csv(fpath, index=False)


def read_single_column_csv_to_string_list(fpath):
    # FIXME: Note dependency with write_list_to_csv (above)
    if not os.path.exists(fpath):
        return []
    else:
        dataframe = pd.read_csv(fpath, dtype=str)
        return list(dataframe[single_column_name])


def set_to_sorted_list(_set):
    return sorted(list(_set))


def fpaths_to_uid(fpaths):
    # TODO: More integrity checks here 
    #       For now simply getting parent directory of the T1 file.
    return os.path.split(os.path.split(fpaths['T1'])[0])[1]


class GANDLFData(object):

    def __init__(self, 
                 data_path=None, 
                 training_batch_size=1,
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
                 handle_data_loss_from_previous_split= True,
                 force_rerun_with_recent_data_loss = True,
                 federated_simulation_train_val_csv_path = None,
                 federated_simulation_institution_name = None,
                 **kwargs):

        self.logger = logging.getLogger('openfl.model_and_data')

        # some hard-coded attributes
        # feature stack order (determines order of feature stack modes)
        # dependency here with mode naming convention used in get_appropriate_file_paths_from_subject_dir
        self.feature_modes = ['T1', 'T2', 'FLAIR', 'T1CE']
        self.label_tag = 'Label'

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

        self.numstring_to_num_headers = {str(name): name for name in self.numeric_header_names.values()}
        # get the header number associated to the subject id as well
        self.numstring_to_num_headers['0'] = 0
        

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
        self.batch_size = training_batch_size
        
        # augmentations apply only for the trianing loader
        self.train_augmentations = data_augmentation

        self.preprocessing = data_preprocessing

        # sudirectories to skip when listing patient subdirectories inside data directory
        self.excluded_subdirs = excluded_subdirs

        # default (otherwise set inside self.get_data_uids_from_disk)
        self.uids_to_subirs = None
        
        #################################################################
        # The following attributes apply only to data-usage='train-val' #
        #################################################################
        
        self.split_instance_dirname = split_instance_dirname
        
        self.shuffle_before_train_val_split = shuffle_before_train_val_split
        self.random_generator_instance = np.random.default_rng(np_split_seed)
        
        self.percent_train = percent_train
        # sanity check
        if (self.percent_train <= 0.0) or (self.percent_train >= 1.0):
            raise ValueError('Value of percent_train must be stricly between 0 and 1.')

        # do we allow new data into training and validation (when some train and val was previously assigned)
        self.allow_new_data_into_previous_split = allow_new_data_into_previous_split

        # do we throw exceptions for missing disk data previously recorded as train or val samples
        # (not recording the new split and missing data info),
        # or record information in a file and do our best to restore percent_train with new samples
        self.handle_data_loss_from_previous_split = handle_data_loss_from_previous_split

        # in the case handle_data_loss_from_previous_split is True and we encounter new missing data, 
        # do we go ahead and record the new split and missing data info, but throw an exception immediately after?
        # (will then provide an alert in the exception message, but a subsequent rerun will avoid the
        # exception as the missing data will not be new anymore)
        self.force_rerun_with_recent_data_loss = force_rerun_with_recent_data_loss

        # do we throw exceptions for data subdirectories with missing files, or just skip them
        self.handle_missing_datafiles = handle_missing_datafiles

        # hard-coded file names and strings
        self.split_info_dirname = 'split_info'
        fets_chall_magic_string = '__USE_DATA_PATH_AS_INSTITUTION_NAME__'

        # Provides a way to define the train and val data directly for all institutions
        #  of a simulated federation from a single csv (no cross-run sanity checks here)
        self.federated_simulation_train_val_csv_path = federated_simulation_train_val_csv_path
        self.federated_simulation_institution_name = federated_simulation_institution_name
        if self.federated_simulation_train_val_csv_path is None:
            if self.federated_simulation_institution_name is not None:
                raise ValueError('federated_simulation_train_val_csv_path needs to be provided when federated_simulation_institution_name is.')
        else:
            if self.federated_simulation_institution_name is None:
                raise ValueError('federated_simulation_institution_name needs to be provided when federated_simulation_train_val_csv_path is.')
            elif self.federated_simulation_institution_name == fets_chall_magic_string:
                if data_path is None:
                    raise ValueError("When federated_simulation_institution_name is set to str(fets_chall_magic_string), data_path must be provided.")
                else:
                    self.federated_simulation_institution_name = data_path          
            elif data_path is not None:
                self.logger.warning('\nfederated_simulation_train_val_csv_path has been provided, so data_path will be ignored.\n')
                    
        #############################################################
        # The above attributes apply only to data-usage='train-val' #
        #############################################################
 
        self.data_path = data_path
        # if federated_simultation_train_val_csv_path is provided, we use it instead of data_path
        if self.federated_simulation_train_val_csv_path is None:
            if self.data_path is None:
                raise ValueError('One of data_path or federated_simulation_train_val_csv_path must be provided.')
            elif not os.path.exists(self.data_path):
                raise ValueError('The provided data path: {} does not exits'.format(self.data_path))
 
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
        self.set_dataframe_headers(self.train_val_headers, list_needed=True)
        if self.federated_simulation_train_val_csv_path is not None:
            self.setup_for_train_val_no_cross_run_state()
        else:
            self.setup_for_train_val_w_cross_run_state()

    def setup_for_train_val_no_cross_run_state(self):
        # load the train val csv (should have info on sample paths for the train and val set of
        # all insitutions of a federated learninng simulation
        train_val_paths = pd.read_csv(self.federated_simulation_train_val_csv_path, dtype=str)
        if 'Partition_ID' not in train_val_paths.columns:
                raise ValueError("The train val csv must contain an 'Partition_ID' column, and it does not.")
        if 'TrainOrVal' not in train_val_paths.columns:
                raise ValueError("The train val csv must contain a 'TrainOrVal' column, and it does not.")
        for header in self.headers_list:
            if (header is not None) and (str(header) not in train_val_paths.columns):
                raise ValueError('The columns of the train val csv must contain all of {} and {} is not present in {}.'.format(self.headers_list, header, train_val_paths.columns))

        # now convert the headers that are numstrings to numbers
        train_val_paths.rename(self.numstring_to_num_headers, axis=1, inplace=True)

        # restrict to the institution provided in the __init__ parameters
        train_val_paths = train_val_paths[train_val_paths['Partition_ID']==self.federated_simulation_institution_name]
        
        train_dataframe = train_val_paths[train_val_paths['TrainOrVal']=='train']
        val_dataframe = train_val_paths[train_val_paths['TrainOrVal']=='val']
        
        # now let's drop all of the columns we do not plan to use
        columns_to_drop = list(set(list(train_val_paths.columns)).difference(set(self.headers_list)))
        train_dataframe.drop(columns_to_drop, axis=1, inplace=True)
        val_dataframe.drop(columns_to_drop, axis=1, inplace=True)

        self.set_train_and_val_loaders(train_dataframe=train_dataframe, val_dataframe=val_dataframe)
        

    def setup_for_train_val_w_cross_run_state(self):
        self.inference_loader = []
            
        # FIXME: below contains some hard coded file names
        self.set_split_info_paths()
        
        past_split_exists = self.sanity_check_split_info()
        if not past_split_exists:
            self.allow_previously_unseen_data = True
        else:
            self.allow_previously_unseen_data = self.allow_new_data_into_previous_split

        on_disk = self.get_data_uids_from_disk()
        self.num_on_disk = len(on_disk)

        past_train, past_val = self.get_past_data()

        lost_train, lost_val = self.compare_disk_to_past(on_disk, past_train, past_val)
   
        init_train = set_to_sorted_list(set(on_disk).intersection(set(past_train)))
        self.num_train_historical_assignments = len(init_train)
        
        init_val = set_to_sorted_list(set(on_disk).intersection(set(past_val)))
        self.num_val_historical_assignments = len(init_val)
        

        if self.allow_previously_unseen_data:
            to_split = set_to_sorted_list(set(on_disk) - set(init_train) - set(init_val))
            self.num_fresh_to_split = len(to_split)
            
            new_train, new_val = self.split(to_split, 
                                            num_existing_train=len(init_train), 
                                            num_existing_val=len(init_val))
            self.num_train_fresh_assignments = len(new_train)
            self.num_val_fresh_assignments = len(new_val)

            train = set_to_sorted_list(set(init_train).union(set(new_train)))    
            val = set_to_sorted_list(set(init_val).union(set(new_val)))          
        else:
            train = init_train
            val = init_val

        self.check_for_undesirable_split(train=train, val=val)

        # writing out lost data and split info (find out if newly lost samples were found)
        newly_lost_train, newly_lost_val = self.record_lost_data(lost_train=lost_train, lost_val=lost_val)
        self.record_split_info(train=train, val=val)

        if (newly_lost_train != [] or newly_lost_val != []) and self.force_rerun_with_recent_data_loss:
            raise ValueError('Train with UIDs: {}, and  Val with UIDs: {} are newly missing, and force_rerun_with_recent_data_loss is True. Put this data back into {} and re-run, or re-run without them.'.format(newly_lost_train, newly_lost_val, self.data_path))
        
        # constructing dataframes, then loaders from the dataframes
        train_dataframe, val_dataframe = self.create_train_val_dataframes(train_uids=train, val_uids=val)     
        self.set_train_and_val_loaders(train_dataframe=train_dataframe, val_dataframe=val_dataframe)

    def get_data_uids_from_disk(self, include_labels=True):
        good_data_uids = []
        uids_to_fpaths = {}
        uids_to_subdirs = {}
        incomplete_data_subdirs = []
        for subdir in self.get_sorted_subdirs():
            subdir_path = os.path.join(self.data_path, subdir)
            fpaths = get_appropriate_file_paths_from_subject_dir(dir_path=subdir_path, 
                                                                 include_labels=include_labels,         
                                                                 handle_missing_datafiles=self.handle_missing_datafiles)
            if fpaths is None:
                if self.handle_missing_datafiles:
                    incomplete_data_subdirs.append(subdir)
                    continue
                else:
                    raise ValueError('Unexpected None return from get_appropriate_file_paths_from_subject_dir')
            else:
                uid = fpaths_to_uid(fpaths)
                good_data_uids.append(uid)
                uids_to_fpaths[uid] = fpaths
                uids_to_subdirs[uid] = subdir
   
        if len(incomplete_data_subdirs) != 0:
            self.logger.debug('\nIgnoring subdirectories: {} as they are missing needed data files.\n'.format(incomplete_data_subdirs))
        
        if hasattr(self, 'uids_to_fpaths'):
            raise RuntimeError('Use-cases comprehended for attribute uids_to_fpaths is currently only write-once.')
        self.uids_to_fpaths = uids_to_fpaths
        if hasattr(self, 'uids_to_subdirs'):
            raise RuntimeError('Use-cases comprehended for attribute uids_to_subdirs is currently only write-once.')
        self.uids_to_subdirs = uids_to_subdirs
        
        return sorted(good_data_uids)

    def get_subdirs_from_uids(self, uids):
        if self.uids_to_subdirs is None:
            raise ValueError('Asking for self.uids_to_subdirs before calling self.get_data_uids_from_disk where this attribute is set.')
        else:
            return [self.uids_to_subdirs[uid] for uid in uids]
                       
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
            need_all = [self.train_info_path, self.val_info_path]
            need_all_or_none = [self.lost_train_info_path, self.lost_val_info_path]
            exists_in_all = [os.path.exists(_path) for _path in need_all]
            exists_in_all_or_none = [os.path.exists(_path) for _path in need_all_or_none]
            if not np.all(exists_in_all):
                raise ValueError('At least one of {} is missing!! Carefully recover, using information produced in model_and_data.log during last run.'.format(need_all))
            if np.any(exists_in_all_or_none) and not np.all(exists_in_all_or_none):
                raise ValueError('All or none of: {} should exists, but exactly one was found!! Carefully recover, using information produced in model_and_data.log during last run.'.format(need_all_or_none))
            past_split_exists = True
        return past_split_exists

    def get_split_info(self):
        self.sanity_check_split_info()
        # returns empty lists if path does not exist
        train_info = read_single_column_csv_to_string_list(fpath=self.train_info_path)
        val_info = read_single_column_csv_to_string_list(fpath=self.val_info_path)
            
        return train_info, val_info

    def get_lost_data(self):
        self.sanity_check_split_info()
        # returns empty lists if path does not exist
        lost_train = read_single_column_csv_to_string_list(fpath=self.lost_train_info_path)
        lost_val = read_single_column_csv_to_string_list(fpath=self.lost_val_info_path)

        return lost_train, lost_val

    def get_past_data(self):
        split_train, split_val = self.get_split_info()
        lost_train, lost_val = self.get_lost_data()
        
        # combine old split with lost data
        past_train = set(split_train).union(set(lost_train))
        past_val = set(split_val).union(set(lost_val))

        # sanity check
        if past_train.intersection(past_val) != set():
            raise ValueError('Somehow there is a record of training intersecting validation.')

        return set_to_sorted_list(past_train), set_to_sorted_list(past_val)

    def compare_disk_to_past(self, on_disk, past_train, past_val):

        lost_train = set_to_sorted_list(set(past_train) - set(on_disk))
        lost_val = set_to_sorted_list(set(past_val) - set(on_disk))

        if (lost_train != []) or (lost_val != []):
            # sanity check
            if not os.path.exists(self.split_instance_dirpath):
                raise RuntimeError('Claiming lost data when no historical data info exists.')
            
            if lost_train != []:
                self.logger.debug('\nWARNING: Training data with UIDs: {} from split: {} now missing on disk.\n'.format(lost_train, self.split_instance_dirname))
                if not self.handle_data_loss_from_previous_split:
                    raise ValueError('Training data with UIDs: {} from split: {} now missing on disk and handle_data_loss_from_previous_split is False.'.format(lost_train, self.split_instance_dirname))
            if lost_val != []:
                self.logger.debug('\nWARNING: Validation data with UIDs: {} from split: {} now missing on disk.\n'.format(lost_val, self.split_instance_dirname))
                if not self.handle_data_loss_from_previous_split:
                    raise ValueError('Validation data with UIDs: {} from split: {} now missing on disk and handle_data_loss_from_previous_split is False.'.format(lost_val, self.split_instance_dirname))

        return lost_train, lost_val           

    def record_split_info(self, train, val):

        self.logger.debug('\n Data split information:')
        self.logger.debug('\n{} good data subdirectories were found in {}'.format(self.num_on_disk, self.data_path))
        if self.num_train_historical_assignments + self.num_val_historical_assignments != 0:
            self.logger.debug('{} of these were known to be used previously, so were assigned to train/val using historical split info.'.format(self.num_train_historical_assignments + self.num_val_historical_assignments))
        if self.allow_previously_unseen_data and self.num_fresh_to_split != 0:
            self.logger.debug('{} of these were previosly unknown, so assigned randomly (with best effort to maintain percent train:{}.'.format(self.num_fresh_to_split, self.percent_train))
        elif not self.allow_previously_unseen_data and (self.num_on_disk > self.num_train_historical_assignments + self.num_val_historical_assignments):
            self.logger.debug('Not allowing previously unseen data samples into split, since old split info exists and allow_new_data_into_previous_split is False.')
        
        self.logger.debug('\nSubdirectories (uids) to be used for training: {}({})'.format(self.get_subdirs_from_uids(train), train))
        if self.num_train_historical_assignments != 0:
            self.logger.debug('{} assigned using previous split info.'.format(self.num_train_historical_assignments))
        if self.allow_previously_unseen_data and (self.num_train_fresh_assignments != 0):
            self.logger.debug('{} newly assigned.'.format(self.num_train_fresh_assignments))


        self.logger.debug('\nSubdirectories (uids) to be used for validation: {}({})'.format(self.get_subdirs_from_uids(val), val))
        if self.num_val_historical_assignments != 0:
            self.logger.debug('{} assigned using previous split info.'.format(self.num_val_historical_assignments))
        if self.allow_previously_unseen_data and (self.num_val_fresh_assignments != 0):
            self.logger.debug('{} newly assigned.\n'.format(self.num_val_fresh_assignments))

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
            write_list_to_single_column_csv(train, fpath=self.train_info_path)
            write_list_to_single_column_csv(val, fpath=self.val_info_path)
    
    def record_lost_data(self, lost_train, lost_val):
        # write lost data info to disk if this information is different from previously recorded. 
        # Also return any lost samples, newly detected on this run.

        lists_empty = (len(lost_train) + len(lost_val) == 0)  
        self.sanity_check_split_info()
        
        # sanity check above ensures either both or none of lost data files are present
        old_lost_info_present = os.path.exists(self.lost_train_info_path)

        newly_lost_train = []
        newly_lost_val = []
        
        if (not lists_empty) or old_lost_info_present:
            self.logger.debug('\nLost data info:\n')
            self.logger.debug('Data with UIDs: {} were previously used for training but now missing from {}'.format(lost_train, self.data_path))
            self.logger.debug('Data with UIDs: {} were previously used for validation but now missing from {}\n'.format(lost_val, self.data_path)) 

            write_out = False
            
            if not old_lost_info_present:
                # here at least one list is not empty and we have no previously recorded lost data
                write_out = True
                newly_lost_train = lost_train
                newly_lost_val = lost_val
            else: 
                # here previous lost info is present, and so we can compare previous with current
                previous_lost_train, previous_lost_val = self.get_lost_data()
                if (previous_lost_train != lost_train) or (previous_lost_val != lost_val):
                    write_out=True
                    newly_lost_train = set_to_sorted_list(set(lost_train) - set(previous_lost_train))
                    newly_lost_val = set_to_sorted_list(set(lost_val) - set(previous_lost_val))

            if write_out:
                if not os.path.exists(self.split_instance_dirpath):
                    os.mkdir(self.split_instance_dirpath)
                write_list_to_single_column_csv(lost_train, fpath=self.lost_train_info_path)
                write_list_to_single_column_csv(lost_val, fpath=self.lost_val_info_path)

        return newly_lost_train, newly_lost_val 

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
        train_info_fname = 'train.csv'
        val_info_fname = 'val.csv'
        lost_train_info_fname = 'lost_train.csv'
        lost_val_info_fname = 'lost_val.csv'

        # derived paths
        split_info_dirpath = os.path.join(self.data_path, self.split_info_dirname)
        self.split_instance_dirpath = os.path.join(split_info_dirpath, self.split_instance_dirname)
        self.train_info_path = os.path.join(self.split_instance_dirpath, train_info_fname)
        self.val_info_path = os.path.join(self.split_instance_dirpath, val_info_fname)
        self.lost_train_info_path = os.path.join(self.split_instance_dirpath, lost_train_info_fname)
        self.lost_val_info_path = os.path.join(self.split_instance_dirpath, lost_val_info_fname)

        pathlib.Path(split_info_dirpath).mkdir(parents=True, exist_ok=True)


    def create_dataframe(self, uids, include_labels):

        columns = {header: [] for header in self.headers_list if (header is not None)}
        for uid in uids:
            fpaths = self.uids_to_fpaths[uid] 
            # write dataframe row
            columns[self.headers_list[0]].append(uid)
            for header in self.headers_list[1:]:
                if header is not None:
                    columns[header].append(fpaths[self.numeric_header_name_to_key[header]])
        return pd.DataFrame(columns)

    def get_sorted_subdirs(self):
        if not os.path.exists(self.data_path):
            raise ValueError('The provided data path: {} does not exits'.format(self.data_path))
        list_dir = os.listdir(self.data_path)
        # filter entries not meant for grabbing subject data
        subdirs_list = np.sort([item for item in list_dir if item not in self.excluded_subdirs and os.path.isdir(os.path.join(self.data_path, item))])

        self.logger.debug("\nFound {} subdirectories under {} excluding the subdirectories that were supposed to be ignored: {}\n".format(len(subdirs_list), self.data_path, self.excluded_subdirs))
        
        return subdirs_list

    def create_train_val_dataframes(self, train_uids, val_uids):
                
        # create the dataframes
        train_dataframe = self.create_dataframe(train_uids, include_labels=True)
        val_dataframe = self.create_dataframe(val_uids, include_labels=True)

        return train_dataframe, val_dataframe

    def setup_for_inference(self):
        self.set_dataframe_headers(self.inference_headers, list_needed=True)
        inference_dataframe = self.create_inference_dataframe()
        
        self.train_loader = []
        self.penalty_loader = []
        self.val_loader = []
        self.inference_loader, _ = self.get_loaders(data_frame=inference_dataframe, train=False, augmentations=None)

    
    def create_inference_dataframe(self):
        inference_uids = self.get_data_uids_from_disk(include_labels=False)
        # create the dataframes
        inference_dataframe = self.create_dataframe(inference_uids, include_labels=False)
    
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

        ## added for reproducibility
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)
        
        g = torch.Generator()
        g.manual_seed(0)
        ## added for reproducibility

        if train:
            loader = DataLoader(data, shuffle=True, batch_size=self.batch_size, worker_init_fn=seed_worker, generator=g)
        else:
            loader = DataLoader(data, shuffle=False, batch_size=1, worker_init_fn=seed_worker, generator=g)
        
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
            companion_loader = DataLoader(companion_data, batch_size=1)

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
        elif self.class_list != ['4', '1||4', '1||2||4']:
           # for fused or trimmed_fused the background is 0 so already set, but otherwise we raise an exception
           raise ValueError('Supporting class list of {} is not present.'.format(self.class_list))
  
        # write in non-background output using the output of cropped features
        output[:, :, small_idx_corner[0]:large_idx_corner[0],small_idx_corner[1]:large_idx_corner[1], small_idx_corner[2]:large_idx_corner[2]] = \
            output_of_cropped[:,:,:large_idx_corner[0]-small_idx_corner[0],:large_idx_corner[1]-small_idx_corner[1],:large_idx_corner[2]-small_idx_corner[2]]
        
        return output

    def write_outputs(self, outputs, dirpath, class_list,  class_axis=1):
        for idx, output in enumerate(outputs):
            fpath = os.path.join(dirpath, "output_" + str(idx) + ".nii.gz")

            # sanity check
            if output.shape[class_axis] != len(class_list):
                    raise ValueError('The provided output does not enumerate classes along axis {} as assumed.'.format(class_axis))
            
            # process float outputs into 0, 1, 2, 4 original labels
            if self.class_list == [0, 1, 2, 4]:
                # here the output should have a multi dim channel enumerating class softmax along class_axis axis
                # check that softmax was used
                if not np.all(np.sum(output, axis=class_axis) - 1 < 1e-6):
                    raise ValueError('The provided output does not appear to have softmax along {} axis as assumed.'.format(class_axis)) 
                # infer label from argmax
                idx_array = np.argmax(output, axis=class_axis)
                new_output = idx_array.apply_(lambda idx : class_list[idx])
            elif self.class_list == ['4', '1||4', '1||2||4']:
                # FIXME: This is one way to infer the original labels (is this the best way?)

                new_shape = [length for idx, length in enumerate(output.shape) if idx != class_axis]
                
                # initializations
                new_output = np.zeros(new_shape)
                slices = [slice(None) for _ in output.shape]

                # write in 4's indicated by ET channel of class_axis
                slices[class_axis] = 0  # 0 is ET channel
                locations_of_4s = output[tuple(slices)]==1  # 1 indicating YES for ET
                new_output[locations_of_4s] = 4

                # write in 1's indicated by TC but not already labeled 4
                slices[class_axis] = 1  # 1 is the TC channel
                locations_of_TC = output[tuple(slices)]==1  # 1 indicating YES for TC
                locations_of_1s = np.logical_and(locations_of_TC, ~locations_of_4s)
                new_output[locations_of_1s] = 1

                # write in 2's indicated by WT but not already labeled 1's or 4's
                slices[class_axis] = 2  # 2 is WT channel
                locations_of_WT = output[tuple(slices)]==1  # 1 indicating YES for WT
                locations_of_1or4 = np.logical_or(locations_of_1s, locations_of_4s)
                locations_of_2s = np.logical_and(locations_of_WT, ~locations_of_1or4)
                new_output[locations_of_2s] = 2

                # sanity check
                np.sum(new_output != 0) == np.sum(np.amax(output, axis=class_axis))
            else:
                raise ValueError('Class list {} not currently supported.'.format(self.class_list))
            
            # shape is currently [1, 155, 240, 240]. for sitk saving we will squeeze
            new_output = new_output[0].transpose([2, 0, 1])
            if list(new_output.shape) != [155, 240, 240]:
                raise ValueError('Unexpected shape {} during processing of output image for sitk savings (was expecting [155, 240, 240]).'.format(new_output.shape))

            # convert array to SimpleITK image 
            image = sitk.GetImageFromArray(new_output)

            self.logger.debug("Writing inference NIfTI image of shape {} to {}".format(new_output.shape, fpath))
            sitk.WriteImage(image, fpath)

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
    



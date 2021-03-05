
# TODO:insert header pointing to GANDLF repo inclusion
# TODO: Should the validation patch sampler be different from the training one?

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


def load_pickle(fname):
    with open(fname, 'rb') as _file:
        return pkl.load(_file)


def dump_pickle(objct, fname):
    with open(fname, 'wb') as _file:
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
                 q_max_length=1,
                 q_num_workers=0,
                 excluded_subdirs = ['log', 'logs', 'split_info'],
                 percent_train = 0.8,
                 in_memory=False,
                 data_usage='train-val',
                 allow_auto_split = False, 
                 shuffle_before_train_val_split=True,
                 allow_new_data_into_preexisting_split = True,
                 allow_disk_data_loss_after_split_creation = True,
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
        self.split_instance_dirname = split_instance_dirname
        # in the event that the split instance directory is absent, do we generate a split or raise and exception?
        self.allow_auto_split = allow_auto_split
        self.shuffle_before_train_val_split = shuffle_before_train_val_split
        self.random_generator_instance = np.random.default_rng(np_split_seed)
        self.percent_train = percent_train
        self.allow_new_data_into_preexisting_split = allow_new_data_into_preexisting_split
        self.allow_disk_data_loss_after_split_creation = allow_disk_data_loss_after_split_creation

        # sudirectories to skip when listing patient subdirectories inside data directory
        self.excluded_subdirs = excluded_subdirs
        
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
            
        # initializations
        all_split_info_present = False

        self.set_headers_and_headers_list(self.train_val_headers, list_needed=True)

        split_info_dirpath = os.path.join(self.data_path, 'split_info')
        if not os.path.exists(split_info_dirpath):
            os.mkdir(split_info_dirpath)
        #  path to directory meant to hold train.csv and val.csv for a specific split
        self.split_instance_dirpath = os.path.join(split_info_dirpath, self.split_instance_dirname)
        self.train_csv_path = os.path.join(self.split_instance_dirpath, 'train.csv')
        self.val_csv_path = os.path.join(self.split_instance_dirpath, 'val.csv')
        self.pickled_split_path = os.path.join(self.split_instance_dirpath, 'pickled_train_val_lists_tuple.pkl')
        self.pickled_lost_training_data_path = os.path.join(self.split_instance_dirpath, 'lost_train.pkl')
        self.pickled_lost_val_data_path = os.path.join(self.split_instance_dirpath, 'lost_val.pkl')

        # check for consistency between train/val csvs and pickled split info, or recover (if possible) from missing files  
        if  os.path.exists(self.split_instance_dirpath):
            # a split under this name exists, is there info in need of recovery?
            csvs_exist = np.all([os.path.exists(path) for path in [self.train_csv_path, self.val_csv_path]])
            pickled_split_exists = os.path.exists(self.pickled_split_path)

            if csvs_exist or pickled_split_exists:
                # here we have enough info, but may need to reproduce the redundant info
                if not csvs_exist:
                    self.recover_csvs_from_pickled_split_info()
                if not pickled_split_exists:
                    print("\nWARNING: The gandlf_data object is recovering missing pickled split info from train and val csvs.\n")
                    self.recover_pickled_split_info_from_csvs()

                all_split_info_present = True
                
            else:
                raise ValueError("A train/val split {} of parent directory {} still exists, but is missing enough info to not be recoverable. Either recover the csvs, or pickled split, or remove {} and allow auto split.".format(self.data_path, self.split_instance_dirname, self.split_instance_dirname)) 

        if all_split_info_present:

            train_subdirs, val_subdirs = self.utilize_existing_split()
            train_dataframe, val_dataframe = self.create_train_val_dataframes(train_subdirs=train_subdirs, val_subdirs=val_subdirs)     
            self.set_train_and_val_loaders(train_dataframe=train_dataframe, val_dataframe=val_dataframe)

            # now that we know none of the subdirs led to missing file exceptions (from ImagesFromDataFrame) we write out the split info
            dataframe_to_string_csv(dataframe=train_dataframe, path=self.train_csv_path)
            dataframe_to_string_csv(dataframe=val_dataframe, path=self.val_csv_path)
            dump_pickle((train_subdirs, val_subdirs), self.pickled_split_path)

        elif self.allow_auto_split:

            train_subdirs, val_subdirs = self.compute_fresh_split()
            train_dataframe, val_dataframe = self.create_train_val_dataframes(train_subdirs=train_subdirs, val_subdirs=val_subdirs)
            self.set_train_and_val_loaders(train_dataframe=train_dataframe, val_dataframe=val_dataframe)

            # now that we know none of the subdirs led to missing file exceptions (from ImagesFromDataFrame) we write out the split info
            os.mkdir(self.split_instance_dirpath)
            dataframe_to_string_csv(dataframe=train_dataframe, path=self.train_csv_path)
            dataframe_to_string_csv(dataframe=val_dataframe, path=self.val_csv_path)
            dump_pickle((train_subdirs, val_subdirs), self.pickled_split_path)
        else:
            raise ValueError('No split under the name of {} is present, and allow_auto_split is False.'.format(self.split_instance_dirname))

    def utilize_existing_split(self):

        # check for consistency of pickled split info against csvs (meanwhile ensuring population of self dataframes)
        csv_train_subdirs, csv_val_subdirs = self.recover_pickled_split_info_from_csvs(to_disk=False)
        pklfile_train_subdirs, pklfile_val_subdirs = load_pickle(self.pickled_split_path)
        if csv_train_subdirs != pklfile_train_subdirs:
            raise ValueError('Train csv and pickled split info do not match (careful review required to ensure we have not introduced train samples into val or vice versa).')
        if csv_val_subdirs != pklfile_val_subdirs:
            raise ValueError('Validation csv and pickled split info do not match (careful review required to ensure we have not introduced train samples into val or vice versa).')

        # checking for data lost from disk since split was produced
    
        disk_subdirs = set(self.get_sorted_subdirs())
        split_train_subdirs = set(csv_train_subdirs)        
        split_val_subdirs = set(csv_val_subdirs)

        these_missing_train = split_train_subdirs - disk_subdirs
        these_missing_val = split_val_subdirs - disk_subdirs

        # this is the new preliminary split (reflects any missing data)
        train_subdirs = split_train_subdirs - these_missing_train
        val_subdirs = split_val_subdirs - these_missing_val

        if (these_missing_train != set([])) or (these_missing_val != set([])):
            
            if these_missing_train != set([]):
                print('\nWARNING: Train samples {} from split {} now missing on disk.\n'.format(these_missing_train, self.split_instance_dirname))
                dump_pickle(list(these_missing_train), self.pickled_lost_training_data_path)
                if not self.allow_disk_data_loss_after_split_creation:
                    raise ValueError('Train samples {} from split {} now missing on disk and allow_disk_data_loss_after_split is False.'.format(these_missing_train, self.split_instance_dirname))
            if these_missing_val != set([]):
                print('\nWARNING: Val samples {} from split {} now missing on disk.\n'.format(these_missing_val, self.split_instance_dirname))
                dump_pickle(list(these_missing_val), self.pickled_lost_val_data_path)
                if not self.allow_disk_data_loss_after_split_creation:
                    raise ValueError('Val samples {} from split {} now missing on disk and allow_disk_data_loss_after_split is False.'.format(these_missing_val, self.split_instance_dirname))
            if self.allow_disk_data_loss_after_split_creation:
                print('\nWriting out new split info to reflect data missing since previous split creation.')
                dump_pickle((list(train_subdirs), list(val_subdirs)), self.pickled_split_path) 
                temp_train_dataframe, temp_val_dataframe = self.create_train_val_dataframes(train_subdirs=train_subdirs, val_subdirs=val_subdirs)
                dataframe_to_string_csv(dataframe=temp_train_dataframe, path=self.train_csv_path)
                dataframe_to_string_csv(dataframe=temp_val_dataframe, path=self.val_csv_path)
         
        if self.allow_new_data_into_preexisting_split:
       
            # recovering information about all missing subdirs (to ensure old val does not become new train and vice versa)
            missing_train = set(load_pickle(self.pickled_lost_training_data_path))
            missing_val = set(load_pickle(self.pickled_lost_val_data_path))

            # we know how to place these subdirs (may be empty)
            add_train_subdirs = list(disk_subdirs.intersection(missing_train)) 
            add_val_subdirs = list(disk_subdirs.intersection(missing_val)) 
             
            # these we will split (trying to maintain self.percent_train best we can) 
            new_subdirs_to_split = list(disk_subdirs - split_train_subdirs - split_val_subdirs - add_train_subdirs - add_val_subdirs)
            # sorting then shuffling for reproducibility with fixed np_split_seed and fixed old split info and new samples
            new_subdirs_to_split = np.sort(new_subdirs_to_split)
            self.random_generator_instance.shuffle(new_subdirs_to_split)

            # one by one, append to train or val according to  
            num_train = len(train_subdirs) + len(add_train_subdirs)
            num_val =  len(val_subdirs) + len(add_val_subdirs)
            total_samples = num_train + num_val
            for subdir in new_subdirs_to_split:
                if float(num_train)/float(total_samples) < self.percent_train:
                    add_train_subdirs.append(subdir)
                else:
                    add_val_subdirs.append(subdir)

            train_subdirs = train_subdirs + add_train_subdirs
            val_subdirs = val_subdirs + add_val_subdirs

        return train_subdirs, val_subdirs
                
    def subdirs_to_subdirpaths(self, subdirs):
        return [os.path.join(self.data_path, subdir) for subdir in subdirs]

    def subdirpaths_to_subdirs(self, subdirpaths):
        return [os.path.split(path)[1] for path in subdirpaths]

    def compute_fresh_split(self):

        # sorting to make process deterministic for a fixed seed
        subdirs = self.get_sorted_subdirs()
        total_subjects = len(subdirs)
        if self.shuffle_before_train_val_split:  
            self.random_generator_instance.shuffle(subdirs)
            # cast back to a list if needed
            subdirs = subdirs.tolist()
            
        # compute the split

        # sanity checks
        if (self.percent_train <= 0.0) or (self.percent_train >= 1.0):
            raise ValueError('Value of percent_train must be stricly between 0 and 1.') 
        split_idx = int(self.percent_train * total_subjects)
        if (split_idx == 0) or (split_idx == total_subjects):
            raise ValueError('Value of percent_train {} is leading to empty val or train set due to only {} subjects found under {} '.format(self.percent_train, total_subjects, self.data_path))
        
        train_subdirs = subdirs[:split_idx]
        val_subdirs = subdirs[split_idx:]
        print('Splitting the {} subjects in {} using percent_train of {}'.format(total_subjects, self.data_path, self.percent_train))
        print('Resulting train and val sets have counts {} and {} respectively.'.format(len(train_subdirs), len(val_subdirs)))

        return train_subdirs, val_subdirs   

    def setup_for_inference(self):
        self.set_headers_and_headers_list(self.inference_headers, list_needed=True)
        inference_dataframe = self.create_inference_dataframe()
        
        self.train_loader = []
        self.penalty_loader = []
        self.val_loader = []
        self.inference_loader, _ = self.get_loaders(data_frame=inference_dataframe, train=False, augmentations=None)
    
    def recover_pickled_split_info_from_csvs(self, to_disk=True):
        temp_train_dataframe = dataframe_as_string(pd.read_csv(self.train_csv_path))
        temp_val_dataframe = dataframe_as_string(pd.read_csv(self.val_csv_path))
        train_subdirs = list(temp_train_dataframe[str(self.train_val_headers['subjectIDHeader'])])
        val_subdirs = list(temp_val_dataframe[str(self.train_val_headers['subjectIDHeader'])])
        subdir_tuple = (train_subdirs, val_subdirs)
        if to_disk:
            dump_pickle(subdir_tuple, self.pickled_split_path)
        else:
            return subdir_tuple

    def recover_csvs_from_pickled_split_info(self):
        print("\nWARNING: The gandlf_data object is recovering missing train and val csvs from pickled split info.\n")
        train_subdirs, val_subdirs = load_pickle(self.pickled_split_path)
        train_dataframe, val_dataframe = self.create_train_val_dataframes(train_subdirs=train_subdirs, val_subdirs=val_subdirs)
        dataframe_to_string_csv(dataframe=train_dataframe, path=self.train_csv_path)
        dataframe_to_string_csv(dataframe=val_dataframe, path=self.val_csv_path)

    def set_train_and_val_loaders(self, train_dataframe, val_dataframe):
        self.train_loader, self.penalty_loader = self.get_loaders(data_frame=train_dataframe, train=True, augmentations=self.train_augmentations)
        self.val_loader, _ = self.get_loaders(data_frame=val_dataframe, train=False, augmentations=None)

        
    def set_headers_and_headers_list(self, headers, list_needed=False):
        self.headers = headers
        if list_needed:
            self.headers_list = [self.headers['subjectIDHeader']] + self.headers['channelHeaders'] + [self.headers['labelHeader']] + self.headers['predictionHeaders']
            

    def create_dataframe_from_subdir_paths(self, subdir_paths, include_labels):

        columns = {header: [] for header in self.headers_list}
        for subdir_path in subdir_paths:
            # grab second to last part of path (subdir name)
            subdir = os.path.split(subdir_path)[1]
            fpaths = get_appropriate_file_paths_from_subject_dir(dir_path=subdir_path, include_labels=include_labels)
            # write dataframe row
            columns[self.headers_list[0]].append(subdir)
            for header in self.headers_list[1:]:
                columns[header].append(fpaths[self.numeric_header_name_to_key[header]])
        return pd.DataFrame(columns)

    def get_sorted_subdirs(self):
        subdirs_list = os.listdir(self.data_path)
        # filter subdirectories not meant for grabbing subject data
        subdirs_list = np.sort([subdir for subdir in subdirs_list if subdir not in self.excluded_subdirs])
        
        return subdirs_list

    def create_train_val_dataframes(self, train_subdirs, val_subdirs):
                
        train_paths = self.subdirs_to_subdirpaths(train_subdirs)
        val_paths = self.subdirs_to_subdirpaths(val_subdirs)
                        
        # create the dataframes
        train_dataframe = self.create_dataframe_from_subdir_paths(train_paths, include_labels=True)
        val_dataframe = self.create_dataframe_from_subdir_paths(val_paths, include_labels=True)

        return train_dataframe, val_dataframe

    
    def create_inference_dataframe(self):
        
        inference_paths = self.subdirs_to_subdirpaths(self.get_sorted_subdirs())
        
        # create the dataframes
        inference_dataframe = self.create_dataframe_from_subdir_paths(inference_paths, include_labels=False)
    
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
    



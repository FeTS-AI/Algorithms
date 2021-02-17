

# TODO:insert header pointing to GANDLF repo inclusion
# TODO: Should the validation patch sampler be different from the training one?

import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
import numpy as np
import torch
import torchio
import pandas as pd

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
        self.label_tag = 'Label'

        # using numerical header names
        self.numeric_header_names = {mode: idx+1 for idx, mode in enumerate(self.feature_modes)}
        self.numeric_header_names[self.label_tag] = len(self.feature_modes) + 1
        # inverse of dictionary above
        self.numeric_header_name_to_key = {value: key for key, value in self.numeric_header_names.items()}
        
        # used as headers for dataframe used to create data loader (when csv's are not provided)
        # dependency (other methods expect the first header to be subject name and subsequent ones being self.feature_modes)
        
        self.default_train_val_headers = {}
        self.default_train_val_headers['subjectIDHeader'] = 0
        self.default_train_val_headers['channelHeaders'] = [self.numeric_header_names[mode] for mode in self.feature_modes]
        self.default_train_val_headers['labelHeader'] = self.numeric_header_names[self.label_tag]
        self.default_train_val_headers['predictionHeaders'] = []
        
        self.default_inference_headers = {}
        self.default_inference_headers['subjectIDHeader'] = 0
        self.default_inference_headers['channelHeaders'] = [self.numeric_header_names[mode] for mode in self.feature_modes]
        self.default_inference_headers['labelHeader'] = None
        self.default_inference_headers['predictionHeaders'] = []
        

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

        if data_usage == 'train-val':
            # get the dataframe and headers
            if isinstance(data_path, dict):
                if ('train' not in data_path) or ('val' not in data_path):
                    raise ValueError('data_path dictionary is missing either train or val key, either privide these, or change to a string data_path (train/val split will then be automatically determined.')
                train_dataframe, train_headers = get_dataframe_and_headers(file_data_full=data_path['train'])
                val_dataframe,  val_headers = get_dataframe_and_headers(file_data_full=data_path['val'])
                
                # validate headers are the same
                for header_type in ['subjectIDHeader', 'channelHeaders', 'labelHeader', 'predictionHeaders']:
                    if train_headers[header_type] != val_headers[header_type]:
                        raise ValueError('Train/Val headers must agree, but found different {} ({} != {})'.format(header_type, train_headers[header_type], val_headers[header_type]))
                self.set_headers_and_headers_list(train_headers)
            else:
                self.set_headers_and_headers_list(self.default_train_val_headers, list_needed=True)
                train_dataframe, val_dataframe = self.create_train_val_dataframes(pardir=data_path, percent_train=percent_train)
            # get the loaders
            self.train_loader, self.penalty_loader = self.get_loaders(data_frame=train_dataframe, train=True, augmentations=self.train_augmentations)
            self.val_loader, _ = self.get_loaders(data_frame=val_dataframe, train=False, augmentations=None)
            self.inference_loader = []

        elif data_usage == 'inference':
            # get the dataframe and headers
            if isinstance(data_path, dict):
                if ('inference' not in data_path):
                    raise ValueError('data_path dictionary is missing the inference key, either privide this entry or change to a string data_path')
                inference_dataframe, self.headers = get_dataframe_and_headers(file_data_full=data_path['inference'])
            else:
                self.set_headers_and_headers_list(self.default_inference_headers, list_needed=True)
                inference_dataframe = self.create_inference_dataframe(pardir=data_path)
            # get the loaders
            self.train_loader = []
            self.penalty_loader = []
            self.val_loader = []
            self.inference_loader, _ = self.get_loaders(data_frame=inference_dataframe, train=False, augmentations=None)
        else:
            raise ValueError('data_usage needs to be either train-val or inference')
        
        self.training_data_size = len(self.train_loader)
        self.validation_data_size = len(self.val_loader)

        
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

    def get_subdir_paths(self, pardir):
        subdirs_list = os.listdir(pardir)
        # filter subdirectories not meant for grabbing subject data
        subdirs_list = [subdir for subdir in subdirs_list if subdir not in self.excluded_subdirs]
        
        # create full paths to subdirs
        subdir_paths_list = [os.path.join(pardir, subdir) for subdir in subdirs_list]
        return subdir_paths_list

    def create_train_val_dataframes(self, pardir, percent_train):
        
        subdir_paths_list = self.get_subdir_paths(pardir)
        total_subjects = len(subdir_paths_list)
        if self.shuffle_before_train_val_split:
            np.random.shuffle(subdir_paths_list)
        
        # compute the split

        # sanity checks
        if (percent_train <= 0.0) or (percent_train >= 1.0):
            raise ValueError('Value of percent_train must be stricly between 0 and 1.') 
        split_idx = int(percent_train * total_subjects)
        if (split_idx == 0) or (split_idx == total_subjects):
            raise ValueError('Value of percent_train {} is leading to empty val or train set due to only {} subjects found under {} '.format(percent_train, total_subjects, pardir))
        
        train_paths = subdir_paths_list[:split_idx]
        val_paths = subdir_paths_list[split_idx:] 
        print('Splitting the {} subjects in {} using percent_train of {}'.format(total_subjects, pardir, percent_train))
        print('Resulting train and val sets have counts {} and {} respectively.'.format(len(train_paths), len(val_paths)))
        
        # create the dataframes
        train_dataframe = self.create_dataframe_from_subdir_paths(train_paths, include_labels=True)
        val_dataframe = self.create_dataframe_from_subdir_paths(val_paths, include_labels=True)

        return train_dataframe, val_dataframe

    
    def create_inference_dataframe(self, pardir):
        
        inference_paths = self.get_subdir_paths(pardir)
        
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
    



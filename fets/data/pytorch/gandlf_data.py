

# TODO:insert header pointing to GANDLF repo inclusion
# TODO: Should the validation patch sampler be different from the training one?

import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235

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

        # extract some parameters from the parameters dict
        self.patch_sampler = parameters['patch_sampler']
        self.psize = parameters['psize']
        # sanity check this patch size will work with the model
        for _dim in self.psize:
            if _dim % divisibility_factor != 0:
               raise ValueError('All dimensions of the patch must be divisible by the model divisibility factor.')

        self.q_max_length = parameters['q_max_length']
        self.q_samples_per_volume = parameters['q_samples_per_volume']
        self.q_num_workers = parameters['q_num_workers']
        self.q_verbose = parameters['q_verbose']
        
        # set attributes that come from parameters
        self.class_list = parameters['model']['class_list']
        self.n_classes = len(self.class_list)
        # There is an assumption of batch size of 1
        self.batch_size = 1

        self.set_augmentation_attributes(parameters=parameters)

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
        else:
            if data_category == 'train':
                train = True
                augmentations = self.train_augmentations
            elif data_category == 'val':
                train = False
                augmentations = self.val_augmentations
            elif data_category == 'inference':
                train = False
                augmentations = self.inference_augmentations
            elif data_category == 'penalty':
                train = False
                augmentations = self.inference_augmentations
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

    def set_augmentation_attributes(self, parameters):

        data_aug_dict = parameters['data_augmentation']

        # we may want to separately specify for train, val, ...
        # if so, all augmentations (train, val, inference) need to be specified
        if 'train' in data_aug_dict:
            if ('val' not in data_aug_dict) or 'inference' not in data_aug_dict:
                raise RuntimeError('If specifying train, val, or inference augmenations, do so for all.')
            self.train_augmentations = data_aug_dict['train']
            self.val_augmentations = data_aug_dict['val']
            self.inference_augmentations = data_aug_dict['inference']
        else:
            self.train_augmentations = data_aug_dict
            self.val_augmentations = data_aug_dict
            self.inference_augmentations = data_aug_dict
        self.preprocessing = parameters['data_preprocessing']

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
        return len(self.train_loader)
    
    def get_validation_data_size(self):
        return len(self.val_loader)
    





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
                 model_params_filepath, 
                 train_augmentations,
                 val_augmentations,
                 inference_augmentations, 
                 preprocessing):


        self.divisibility_factor = divisibility_factor
        
        parameters = parseConfig(model_params_filepath)

        # extract some parameters from the parameters dict
        patch_sampler = parameters['patch_sampler']
        psize = parameters['psize']
        # sanity check this patch size will work with the model
        for _dim in psize:
            if _dim % divisibility_factor != 0:
               raise ValueError('All dimensions of the patch must be divisible by the model divisibility factor.')

        q_max_length = parameters['q_max_length']
        q_samples_per_volume = parameters['q_samples_per_volume']
        q_num_workers = parameters['q_num_workers']
        q_verbose = parameters['q_verbose']
        if isinstance(parameters['data_augmentation'], dict):
            train_augmentations = parameters['data_augmentation']['train']
            val_augmentations = parameters['data_augmentation']['val']
            inference_augmentations = parameters['data_augmentation']['inference']
        else:
            train_augmentations = parameters['data_augmentation']
            val_augmentations = parameters['data_augmentation']
            inference_augmentations = parameters['data_augmentation']
        preprocessing = parameters['data_preprocessing']
        batch_size = parameters['batch_size']
        
        # data_path here is a dict (here filling in None values where appropriate)
        for data_type in ['train', 'val', 'infernce', 'penalty']:
            if data_type not in data_path:
                data_path[data_path] = None

        # get the training data if the path is provided
        if data_path['train'] is None:
            self.train_loader = []
        else:
            trainingDataFromPickle, train_headers = get_dataframe_and_headers(file_data_full=data_path['train'])
            trainingDataForTorch = ImagesFromDataFrame(dataframe=trainingDataFromPickle, 
                                                       psize=psize, 
                                                       headers=train_headers, 
                                                       q_max_length=q_max_length, 
                                                       q_samples_per_volume=q_samples_per_volume,
                                                       q_num_workers=q_num_workers, 
                                                       q_verbose=q_verbose, 
                                                       sampler=patch_sampler, 
                                                       train=True, 
                                                       augmentations=train_augmentations, 
                                                       preprocessing = preprocessing)
            self.train_loader = DataLoader(trainingDataForTorch, batch_size=batch_size, shuffle=True)

        # get the validation data if the path is provided
        if data_path['val'] is None:
            self.val_loader = []
        else:
            validationDataFromPickle, val_headers = get_dataframe_and_headers(file_data_full=data_path['val'])
            validationDataForTorch = ImagesFromDataFrame(dataframe=validationDataFromPickle, 
                                                         psize=psize, 
                                                         headers=val_headers, 
                                                         q_max_length=q_max_length, 
                                                         q_samples_per_volume=q_samples_per_volume,
                                                         q_num_workers=q_num_workers, 
                                                         q_verbose=q_verbose, 
                                                         sampler=patch_sampler, 
                                                         train=False, 
                                                         augmentations=val_augmentations, 
                                                         preprocessing=preprocessing) # may or may not need to add augmentations here
            self.val_loader = DataLoader(validationDataForTorch, batch_size=batch_size)
        
        
        # get the inference data if the path is provided
        if data_path['inference'] is None:
            self.inference_loader = []
        else:
            inferenceDataFromPickle, inference_headers = get_dataframe_and_headers(file_data_full=data_path['inference'])
            inferenceDataForTorch = ImagesFromDataFrame(dataframe=inferenceDataFromPickle,
                                                        psize=psize,
                                                        headers=inference_headers, 
                                                        q_max_length=q_max_length, 
                                                        q_samples_per_volume=q_samples_per_volume, 
                                                        q_num_workers=q_num_workers, 
                                                        q_verbose=q_verbose, 
                                                        sampler=patch_sampler, 
                                                        train=False, 
                                                        augmentations=inference_augmentations, 
                                                        preprocessing=preprocessing)
            self.inference_loader = DataLoader(inferenceDataForTorch, batch_size=batch_size)
        
        # get the penalty data if the path is provided
        if data_path['penalty'] is None:
            self.penalty_loader = []
        else:
            # TODO: No augmentations for penalty data?
            penaltyDataFromPickle, penalty_headers = get_dataframe_and_headers(file_data_full=data_path['penalty'])   
            penaltyData = ImagesFromDataFrame(dataframe=penaltyDataFromPickle, 
                                              psize=psize, 
                                              headers=penalty_headers, 
                                              q_max_length=q_max_length, 
                                              q_samples_per_volume=q_samples_per_volume, 
                                              q_num_workers=q_num_workers, 
                                              q_verbose=q_verbose, 
                                              sampler = patch_sampler, 
                                              train=False, 
                                              augmentations=None,
                                              preprocessing=preprocessing) 
            #TODO: Dependency on train loader? (notice both are shuffling)
            self.penalty_loader = DataLoader(penaltyData, batch_size=batch_size, shuffle=True)

        self.training_data_size = len(self.train_loader)
        self.validation_data_size = len(self.val_loader)

        def get_train_loader(self):
            return self.train_loader
        
        def get_val_loader(self):
            return self.val_loader

        def get_inference_loader(self):
            return self.inference_loader

        def get_penalty_loader(self):
            return self.penalty_loader
        



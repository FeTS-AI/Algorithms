

# TODO:insert header pointing to GANDLF repo inclusion
# TODO: Should the validation patch sampler be different from the training one?

import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235

from torch.utils.data import DataLoader

from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame

from fets.data.gandlf_utils import get_dataframe_and_headers

class GANDLFData(object):

    def __init__(self, 
                 file_data_full, 
                 batch_size, 
                 psize, 
                 q_max_length, 
                 q_samples_per_volume, 
                 q_num_workers, 
                 q_verbose, 
                 patch_sampler, 
                 train_augmentations,
                 val_augmentations,
                 inference_augmentations, 
                 preprocessing):
        self.training_batch_size = batch_size
        self.psize = psize
        self.q_max_length = q_max_length 
        self.q_samples_per_volume = q_samples_per_volume
        self.q_num_workers = q_num_workers
        self.q_verbose = q_verbose
        self.sampler = patch_sampler
        self.train_augmentations = train_augmentations
        self.val_augmentations = val_augmentations
        self.inference_augmentations = inference_augmentations
        self.preprocessing = preprocessing


        # TODO How does this relate to the trainingDataFromPickle etc.
        data_full, headers = get_dataframe_and_headers(file_data_full)

        # Setting up the training loader
        trainingDataForTorch = ImagesFromDataFrame(dataframe=trainingDataFromPickle, 
                                                   psize=psize, 
                                                   headers=headers, 
                                                   q_max_length=q_max_length, 
                                                   q_samples_per_volume=q_samples_per_volume,
                                                   q_num_workers=q_num_workers, 
                                                   q_verbose=q_verbose, 
                                                   sampler=patch_sampler, 
                                                   train=True, 
                                                   augmentations=train_augmentations, 
                                                   preprocessing = preprocessing)
        train_loader = DataLoader(trainingDataForTorch, batch_size=batch_size, shuffle=True)

        # Setting up the validation loader
        validationDataForTorch = ImagesFromDataFrame(dataframe=validationDataFromPickle, 
                                                     psize=psize, 
                                                     headers=headers, 
                                                     q_max_length=q_max_length, 
                                                     q_samples_per_volume=q_samples_per_volume,
                                                     q_num_workers=q_num_workers, 
                                                     q_verbose=q_verbose, 
                                                     sampler=patch_sampler, 
                                                     train=False, 
                                                     augmentations=val_augmentations, 
                                                     preprocessing=preprocessing) # may or may not need to add augmentations here
        val_loader = DataLoader(validationDataForTorch, batch_size=1)
        

        # Setting up the inference loader
        inferenceDataForTorch = ImagesFromDataFrame(dataframe=self.inferenceDataFromPickle,
                                                    psize=self.psize,
                                                    headers=self.headers, 
                                                    q_max_length=self.q_max_length, 
                                                    q_samples_per_volume=self.q_samples_per_volume, 
                                                    q_num_workers=self.q_num_workers, 
                                                    q_verbose=self.q_verbose, 
                                                    sampler=patch_sampler, 
                                                    train=False, 
                                                    augmentations=self.augmentations, 
                                                    preprocessing=self.preprocessing)
        self.inference_loader = DataLoader(inferenceDataForTorch, batch_size=self.batch_size)

        # define a seaparate data loader for penalty calculations
        penaltyData = ImagesFromDataFrame(dataframe=trainingDataFromPickle, 
                                          psize=psize, 
                                          headers=headers, 
                                          q_max_length=q_max_length, 
                                          q_samples_per_volume=q_samples_per_volume, 
                                          q_num_workers=q_num_workers, 
                                          q_verbose=q_verbose, 
                                          sampler = patch_sampler, 
                                          train=False, 
                                          augmentations=None,
                                          preprocessing=preprocessing) 
        #TODO: Dependency on train loader? (notice both are shuffling)
        penalty_loader = DataLoader(penaltyData, batch_size=batch_size, shuffle=True)

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
        



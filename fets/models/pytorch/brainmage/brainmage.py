# The following code is modified from https://github.com/CBICA/BrainMaGe which has the following license:

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
import time
import sys, os
import ast
import tqdm
import math
from itertools import product

import pandas as pd
import random

import torchio
import torch
import torch.optim as optim
from torch.autograd import Variable

from GANDLF.utils import one_hot

from openfl import load_yaml
from openfl.models.pytorch import PyTorchFLModel
from .losses import MCD_loss, DCCE, CE, MCD_MSE_loss, dice_loss, average_dice_over_channels

# TODO: Run in CONTINUE_LOCAL or RESET optimizer modes for now, later ensure that the cyclic learning rate is properly handled for CONTINUE_GLOBAL.
# FIXME: do we really want to keep loss at 1-dice rather than -ln(dice)
# FIXME: Turn on data augmentation for training (double check all)


 # TODO: Temporarily using patching in the model code (until GANDLF patching is plugged in) 
def random_slices(array, psize):
    # an example expected shape is: (1, 1, 240, 240, 155)
    # the patch will not apply to the first two dimensions
    shape = array.shape[2:]
    slices = [slice(None), slice(None)]
    for axis, length in enumerate(psize):
        if shape[axis] > length:
            shift = random.randint(0,shape[axis] - length)
            slices.append(slice(shift, shift + length))
    return slices


def crop(array, slices):
    return array[tuple(slices)]


def cyclical_lr(stepsize, min_lr, max_lr):
    #Scaler : we can adapt this if we do not want the triangular LR
    scaler = lambda x:1
    #Lambda function to calculate the LR
    lr_lambda = lambda it: max_lr - (max_lr - min_lr)*relative(it,stepsize)
    #Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1+it/(2*stepsize))
        x = abs(it/stepsize - 2*cycle + 1)
        return max(0,(1-x))*scaler(cycle)
    return lr_lambda


class BrainMaGeModel(PyTorchFLModel):

    def __init__(self, 
                 data, 
                 base_filters, 
                 learning_rate, 
                 loss_function, 
                 opt, 
                 device='cpu',
                 n_classes=2,
                 n_channels=4,
                 psize=[128,128,128],
                 smooth=1e-7,
                 use_penalties=False,
                 **kwargs):
        super().__init__(data=data, device=device, **kwargs)
        
        self.device = device

        # FIXME: this puts priority for these values on data object over flplan. Is this correct?
        if hasattr(data, 'n_classes') and data.n_classes is not None:
            self.n_classes = data.n_classes
        else:
            self.n_classes = n_classes

        if hasattr(data, 'n_channels') and data.n_channels is not None:
            self.n_channels = data.n_channels
        else:
            self.n_channels = n_channels

        if hasattr(data, 'psize') and data.psize is not None:
            self.psize = data.psize
        else:
            self.psize = psize

        # setting parameters as attributes
        # training parameters
        self.learning_rate = learning_rate
        self.which_loss = loss_function
        self.opt = opt
        # model parameters
        self.binary_classification = self.n_classes == 2
        if self.binary_classification:
            self.label_channels = 1
        else:
            self.label_channels = self.n_classes
        self.base_filters = base_filters
        self.smooth = smooth
        self.which_model = self.__repr__()
        self.use_panalties = use_penalties
        
        ############### CHOOSING THE LOSS FUNCTION ###################
        if self.which_loss == 'dc':
            self.loss_fn  = MCD_loss
        elif self.which_loss == 'dcce':
            self.loss_fn  = DCCE
        elif self.which_loss == 'ce':
            self.loss_fn = CE
        elif self.which_loss == 'mse':
            self.loss_fn = MCD_MSE_loss
        else:
            raise ValueError('{} loss is not supported'.format(self.which_loss))

        # TODO: remove print statements here
        print("Computing channel keys and train paths.")
        self.channel_keys, self.train_paths = self.get_channel_keys_and_train_paths()
        
        self.dice_penalty_dict = None
        if self.use_panalties:
            # prepare penalties dict
            _, self.dice_penalty_dict = self.prep_penalties()

    def init_optimizer(self):
        if self.opt == 'sgd':
            self.optimizer = optim.SGD(self.parameters(),
                                       lr= self.learning_rate,
                                       momentum = 0.9)
        if self.opt == 'adam':    
            self.optimizer = optim.Adam(self.parameters(), 
                                        lr = self.learning_rate, 
                                        betas = (0.9,0.999), 
                                        weight_decay = 0.00005)

        # TODO: To sync learning rate cycle, we assume single epoch per round and that the data loader is allowing partial batches!!!
        step_size = int(np.ceil(self.data.get_training_data_size()/self.data.batch_size))
        if step_size == 0:
            # This only happens when we have no training data so will not be using the optimizer
            step_size = 1
            print("\nNo training data is present, so cyclic optimizer being set with step size of 1.\n")
        clr = cyclical_lr(step_size, min_lr = 0.000001, max_lr = 0.001)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, [clr])
        sys.stdout.flush()
    
    def reset_opt_vars(self, **kwargs):
        self.init_optimizer(**kwargs)

    # The following will all be implemented by the child class
    def init_network(self, device, **kwargs):
        raise NotImplementedError()
    
    def forward(self, x):
        raise NotImplementedError()

    def get_channel_keys_and_train_paths(self):
        # Getting the channels for training and removing all the non numeric entries from the channels
        train_paths = []
        for subject in self.data.get_train_loader():
            # Example subject keys: ['0', '1', '2', '3', 'label', 'index_ini']
            # Example subject['0'] keys are: ['data', 'affine', 'path', 'stem', 'type']
            train_paths.append(subject['0']['path'])
          
            # use last subject to inspect channel keys
            channel_keys = []
            for key in subject.keys():
                if key.isnumeric():
                    channel_keys.append(key)

            return channel_keys, train_paths
        return None, None

    def prep_penalties(self):

        # initialize without considering background
        dice_weights_dict = {} # average for "weighted averaging"
        dice_penalty_dict = {} # penalty for misclassification
        for i in range(1, self.n_classes):
            dice_weights_dict[i] = 0
            dice_penalty_dict[i] = 0

        penalty_loader = self.data.get_penalty_loader()
        
        # get the weights for use for dice loss
        total_nonZeroVoxels = 0
        
        # dice penalty is calculated on the basis of the masks (processed here) and predicted labels
        # iterate through full data (may differ from training data by not being cropped for example)
        for subject in penalty_loader: 
            # accumulate dice weights for each label
            mask = subject['label'][torchio.DATA]
            one_hot_mask = one_hot(mask, self.data.class_list)
            for i in range(1, self.n_classes):
                currentNumber = torch.nonzero(one_hot_mask[:,i,:,:,:], as_tuple=False).size(0)
                dice_weights_dict[i] = dice_weights_dict[i] + currentNumber # class-specific non-zero voxels
                total_nonZeroVoxels = total_nonZeroVoxels + currentNumber # total number of non-zero voxels to be considered

        if total_nonZeroVoxels == 0:
            raise RuntimeError('Trying to train on data where every label mask is background class only.')

        # get the penalty values - dice_weights contains the overall number for each class in the training data
        for i in range(1, self.n_classes):
            penalty = total_nonZeroVoxels # start with the assumption that all the non-zero voxels make up the penalty
            for j in range(1, self.n_classes):
                if i != j: # for differing classes, subtract the number
                    penalty = penalty - dice_penalty_dict[j]
            
            dice_penalty_dict[i] = penalty / total_nonZeroVoxels # this is to be used to weight the loss function
        dice_weights_dict[i] = 1 - dice_weights_dict[i]# this can be used for weighted averaging

        return dice_weights_dict, dice_penalty_dict

    def train_batches(self, num_batches, use_tqdm=False):
        num_subjects = num_batches
        
        device = torch.device(self.device)

        ################################ PRINTING SOME STUFF ######################
        print("\nHostname   :" + str(os.getenv("HOSTNAME")))
        sys.stdout.flush()

        print("Training Data Samples: ", len(self.data.train_loader.dataset))
        sys.stdout.flush()

        print('Using device:', device)
        if device.type == 'cuda':
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1),
                'GB')
            print('Cached: ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

        sys.stdout.flush()

        train_loader = self.data.get_train_loader()

        if train_loader == []:
            raise RuntimeError("Attempting to run training with an empty training loader.")

        if use_tqdm:
            train_loader = tqdm.tqdm(train_loader, desc="training for this round")

        total_round_training_loss = 0
        subject_num = 0

        # set to "training" mode
        self.train()
        while subject_num < num_subjects:
                       
            total_loss = 0
            for subject in train_loader:
                if subject_num >= num_subjects:
                    break
                else:
                    # Load the subject and its ground truth
                    # this is when we are using pt_brainmagedata
                    if ('features' in subject.keys()) and ('gt' in subject.keys()):
                        features = subject['features']
                        mask = subject['gt']
                    # this is when we are using gandlf loader   
                    else:
                        features = torch.cat([subject[key][torchio.DATA] for key in self.channel_keys], dim=1)
                        mask = subject['label'][torchio.DATA]

                        # TODO: temporarily patching here (should be done in loader instead)
                        slices = random_slices(mask, self.data.psize)
                        mask = crop(mask, slices=slices)
                        # for the feature array, we skip the first axis as it enumerates the modalities
                        features = crop(features, slices=slices)

                        mask = one_hot(mask, self.data.class_list)
                        
                    # Loading features into device
                    features, mask = features.float().to(device), mask.float().to(device)
                    # TODO: Variable class is deprecated - parameteters to be given are the tensor, whether it requires grad and the function that created it   
                    features, mask = Variable(features, requires_grad = True), Variable(mask, requires_grad = True)
                    # Making sure that the optimizer has been reset
                    self.optimizer.zero_grad()
                    # Forward Propagation to get the output from the models
                    
                    # TODO: Not recommended? (https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/6)will try without
                    #torch.cuda.empty_cache()
                    
                    output = self(features.float())
                    # Computing the loss
                    loss = self.loss_fn(output.double(), mask.double(),num_class=self.label_channels, weights=self.dice_penalty_dict)
                    # Back Propagation for model to learn
                    loss.backward()
                    #Updating the weight values
                    self.optimizer.step()
                    #Pushing the dice to the cpu and only taking its value
                    loss.cpu().data.item()
                    total_loss += loss
                    self.lr_scheduler.step()

                    # TODO: Not recommended? (https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/6)will try without
                    #torch.cuda.empty_cache()

                    subject_num += 1

            total_round_training_loss += total_loss
        # we return the average batch loss over all epochs trained this round
        return total_round_training_loss / num_subjects

    def validate(self, use_tqdm=False):
        device = torch.device(self.device)       
        self.eval()
        
        total_dice = 0
        
        val_loader = self.data.get_val_loader()

        if val_loader == []:
            raise RuntimeError("Attempting to run validation with an empty val loader.")

        if use_tqdm:
            val_loader = tqdm.tqdm(val_loader, desc="validate")

        for subject in val_loader:
            with torch.no_grad():
                # this is when we are using pt_brainmagedata
                if ('features' in subject.keys()) and ('gt' in subject.keys()):
                    features = subject['features']
                    mask = subject['gt']
                # this is when we are using gandlf loader   
                else:
                    features = torch.cat([subject[key][torchio.DATA] for key in self.channel_keys], dim=1)
                    mask = subject['label'][torchio.DATA]

                    # TODO: temporarily patching here (should support through the loader instead)
                    slices = random_slices(mask, self.data.psize)
                    mask = crop(mask, slices=slices)
                    # for the feature array, we skip the first axis as it enumerates the modalities
                    features = crop(features, slices=slices)
                    
                    mask = one_hot(mask, self.data.class_list)
                    
                features, mask = features.to(device), mask.to(device)
                output = self(features.float())
                curr_dice = average_dice_over_channels(output.double(), mask.double(), self.binary_classification).cpu().data.item()
                total_dice += curr_dice
        #Computing the average dice
        average_dice = total_dice/len(val_loader)

        return average_dice


    

    

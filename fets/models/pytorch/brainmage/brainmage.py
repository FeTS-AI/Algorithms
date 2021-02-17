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

import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
import torchio

import numpy as np
import time
import sys, os
import ast
import tqdm
import math
from itertools import product

import pandas as pd
import random
from copy import deepcopy

import torchio
import torch
import torch.optim as optim
from torch.autograd import Variable

from GANDLF.utils import one_hot

from openfl import load_yaml
from openfl.models.pytorch import PyTorchFLModel
from .losses import MCD_loss, DCCE, CE, MCD_MSE_loss, dice_loss, average_dice_over_channels, clinical_dice_loss, clinical_dice_log_loss, clinical_dice

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


def cyclical_lr(cycle_length, min_lr_multiplier, max_lr_multiplier):
    # Lambda function to calculate what to multiply the inital learning rate by
    # The beginning and end of the cycle result in highest multipliers (lowest at the center)
    mult = lambda it: max_lr_multiplier * rel_dist(it, cycle_length) + min_lr_multiplier * (1 - rel_dist(it, cycle_length))
    
    def rel_dist(iteration, cycle_length):
        # relative_distance from iteration to the center of the cycle
        # equal to 1 at beggining of cycle and 0 right at the cycle center

        # reduce the iteration to less than the cycle length
        iteration = iteration % cycle_length
        return 2 * abs(iteration - cycle_length/2.0) / cycle_length

    return mult


class BrainMaGeModel(PyTorchFLModel):

    def __init__(self, 
                 data, 
                 base_filters, 
                 min_learning_rate, 
                 max_learning_rate,
                 learning_rate_cycles_per_epoch,
                 loss_function, 
                 opt, 
                 device='cpu',
                 n_classes=4,
                 n_channels=4,
                 psize=[128,128,128],
                 smooth=1e-7,
                 use_penalties=False, 
                 infer_gandlf_images_with_cropping = False,
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

        # we currently have a hard coded triangular learning rate scheduler
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.learning_rate_cycles_per_epoch = learning_rate_cycles_per_epoch

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

        # used only when using the gandlf_data object
        # (will we crop external zero-planes, infer, then pad output with zeros OR
        #  get outputs for multiple patches - fusing the outputs)
        self.infer_gandlf_images_with_cropping = infer_gandlf_images_with_cropping
        
        ############### CHOOSING THE LOSS FUNCTION ###################
        if self.which_loss == 'dc':
            self.loss_fn  = MCD_loss
        elif self.which_loss == 'dcce':
            self.loss_fn  = DCCE
        elif self.which_loss == 'ce':
            self.loss_fn = CE
        elif self.which_loss == 'mse':
            self.loss_fn = MCD_MSE_loss
        elif self.which_loss == 'cdl':
            self.loss_fn = clinical_dice_loss
        elif self.which_loss == 'cdll':
            self.loss_fn = clinical_dice_log_loss
        else:
            raise ValueError('{} loss is not supported'.format(self.which_loss))

        self.channel_keys= self.get_channel_keys()
        
        self.dice_penalty_dict = None
        if self.use_panalties:
            # prepare penalties dict
            _, self.dice_penalty_dict = self.prep_penalties()

    def init_optimizer(self):
        if self.opt == 'sgd':
            self.optimizer = optim.SGD(self.parameters(),
                                       lr= self.max_learning_rate,
                                       momentum = 0.9)
        if self.opt == 'adam':    
            self.optimizer = optim.Adam(self.parameters(), 
                                        lr = self.max_learning_rate, 
                                        betas = (0.9,0.999), 
                                        weight_decay = 0.00005)

        # If this is removed, need to redo the cylce_length calculation below
        assert self.data.batch_size == 1

        # TODO: To sync learning rate cycle across collaborators, we assume each collaborator is training 
        # a set fraction of an epoch (rather than a set number of batches) otherwise use batch_num based cycle length
        cycle_length =  int(float(self.data.get_training_data_size()) / float(self.learning_rate_cycles_per_epoch))
        if cycle_length == 0:
            if self.data.get_training_data_size == 0:
                cycle_length = 1
                print("\nNo training data is present, so setting silly cyclic length for scheduler.\n")
            else:
                raise ValueError("learning_rate_cycles_per_epoch is set to {} which cannot be greater than the number of training samples (which is {}).".format(self.learning_rate_cycles_per_epoch, self.data.get_training_data_size()))

        # inital learning rates (see above) are set to the max learning rate value
        # scheduling is performed thereafter by multiplying the optimizer rates
        min_lr_multiplier = float(self.min_learning_rate) / float(self.max_learning_rate)
        max_lr_multiplier = 1.0
        clr = cyclical_lr(cycle_length=cycle_length, 
                          min_lr_multiplier = min_lr_multiplier, 
                          max_lr_multiplier = max_lr_multiplier)
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, [clr])
    
    def reset_opt_vars(self, **kwargs):
        self.init_optimizer(**kwargs)

    # The following will all be implemented by the child class
    def init_network(self, device, **kwargs):
        raise NotImplementedError()
    
    def forward(self, x):
        raise NotImplementedError()

    def get_channel_keys(self):
        # Getting one training subject
        channel_keys = []
        for subject in self.data.get_train_loader():
            # break
          
            # use last subject to inspect channel keys
            # channel_keys = []
            for key in subject.keys():
                if key.isnumeric():
                    channel_keys.append(key)

            return channel_keys
        return channel_keys

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

        # dice_weights_dict_temp = deepcopy(dice_weights_dict)
        dice_weights_dict = {k: (v / total_nonZeroVoxels) for k, v in dice_weights_dict.items()} # divide each dice value by total nonzero
        dice_penalty_dict = deepcopy(dice_weights_dict) # deep copy so that both values are preserved
        dice_penalty_dict = {k: 1 - v for k, v in dice_weights_dict.items()} # subtract from 1 for penalty
        total = sum(dice_penalty_dict.values())
        dice_penalty_dict = {k: v / total for k, v in dice_penalty_dict.items()} # normalize penalty to ensure sum of 1
        # dice_penalty_dict = get_class_imbalance_weights(trainingDataFromPickle, parameters, headers, is_regression, class_list) # this doesn't work because ImagesFromDataFrame gets import twice, causing a "'module' object is not callable" error

        return dice_weights_dict, dice_penalty_dict

    def infer_batch_with_no_numpy_conversion(self, features, **kwargs):
        """Very similar to base model infer_batch, but does not
           explicitly convert the output to numpy.
           Run inference on a batch
        Args:
            features: Input for batch
        Gets the outputs for the inputs provided.
        """

        device = torch.device(self.device)
        self.eval()

        with torch.no_grad():
            features = features.to(device)
            output = self(features.float())
            output = output.cpu()
        return output

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

        total_loss = 0
        subject_num = 0
        num_nan_losses = 0

        # set to "training" mode
        self.train()
        while subject_num < num_subjects:
                       
            for subject in train_loader:
                if subject_num >= num_subjects:
                    break
                else:
                    if device.type == 'cuda':
                        print('=== Memory (allocated; cached) : ', round(torch.cuda.memory_allocated(0)/1024**3, 1), '; ', round(torch.cuda.memory_reserved(0)/1024**3, 1))
                    # Load the subject and its ground truth
                    # this is when we are using pt_brainmagedata
                    if ('features' in subject.keys()) and ('gt' in subject.keys()):
                        features = subject['features']
                        mask = subject['gt']
                    # this is when we are using gandlf loader   
                    else:
                        features = torch.cat([subject[key][torchio.DATA] for key in self.channel_keys], dim=1)
                        mask = subject['label'][torchio.DATA]

                    print("\n\nTrain features with shape: {}\n".format(features.shape))

                    mask = one_hot(mask, self.data.class_list)
                        
                    # Loading features into device
                    features, mask = features.float().to(device), mask.float().to(device)
                    # TODO: Variable class is deprecated - parameters to be given are the tensor, whether it requires grad and the function that created it   
                    # features, mask = Variable(features, requires_grad = True), Variable(mask, requires_grad = True)
                    # Making sure that the optimizer has been reset
                    self.optimizer.zero_grad()
                    # Forward Propagation to get the output from the models
                    
                    # TODO: Not recommended? (https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/6)will try without
                    #torch.cuda.empty_cache()
                    
                    output = self(features.float())
                    # Computing the loss
                    loss = self.loss_fn(output.float(), mask.float(),num_classes=self.label_channels, weights=self.dice_penalty_dict, class_list=self.data.class_list)
                    # Back Propagation for model to learn (unless loss is nan)
                    if torch.isnan(loss):
                        num_nan_losses += 1
                    else:
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

        num_subject_grads = num_subjects - num_nan_losses

        # we return the average batch loss over all epochs trained this round (excluding the nan results)
        # we also return the number of samples that produced nan losses, as well as total samples used
        # FIXME: In a federation we may want the collaborators data size to be modified when backprop is skipped.
        return {"loss": total_loss / num_subject_grads, "num_nan_losses": num_nan_losses, "num_samples_used": num_subjects }

    def validate(self, use_tqdm=False):
        
        total_dice = 0
        
        val_loader = self.data.get_val_loader()

        if val_loader == []:
            raise RuntimeError("Attempting to run validation with an empty val loader.")

        if use_tqdm:
            val_loader = tqdm.tqdm(val_loader, desc="validate")

        for subject in val_loader:
            # this is when we are using pt_brainmagedata
            if ('features' in subject.keys()) and ('gt' in subject.keys()):
                features = subject['features']
                mask = subject['gt']
        
                output = self.infer_batch_with_no_numpy_conversion(features=features)
                    
            # using the gandlf loader   
            else:
                features = torch.cat([subject[key][torchio.DATA] for key in self.channel_keys], dim=1)
                mask = subject['label'][torchio.DATA]

                if self.infer_gandlf_images_with_cropping:
                    output = self.data.infer_with_crop(model_inference_function=[self.infer_batch_with_no_numpy_conversion], 
                                                        features=features)
                else:
                    output = self.data.infer_with_crop_and_patches(model_inference_function=[self.infer_batch_with_no_numpy_conversion], 
                                                                    features=features)

                    
                
            # one-hot encoding of ground truth
            mask = one_hot(mask, self.data.class_list)
            
            # sanity check that the output and mask have the same shape
            if output.shape != mask.shape:
                raise ValueError('Model output and ground truth mask are not the same shape.')

            # curr_dice = average_dice_over_channels(output.float(), mask.float(), self.binary_classification).cpu().data.item()
            curr_dice = clinical_dice(output.float(), mask.float(), class_list=self.data.class_list).cpu().data.item()
            total_dice += curr_dice
                
        #Computing the average dice
        average_dice = total_dice/len(val_loader)

        return average_dice


    

    

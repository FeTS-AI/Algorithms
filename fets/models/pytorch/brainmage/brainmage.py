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

import logging
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
from .losses import MCD_loss, MCD_MSE_loss, dice_loss
from .losses import brats_dice_loss, brats_dice_log_loss, brats_dice, brats_dice_loss_w_background, brats_dice_loss_w_crossentropy
from .losses import background_dice_loss, crossentropy, dice_loss_skipping_first_channel, dice_loss_all_channels, mirrored_brats_dice_loss
from .losses import fets_phase2_validation

# TODO: Run in CONTINUE_LOCAL or RESET optimizer modes for now, later ensure that the cyclic learning rate is properly handled for CONTINUE_GLOBAL.
# FIXME: do we really want to keep loss at 1-dice rather than -ln(dice)


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


def nan_check(tensor, tensor_description):
    tensor = tensor.cpu()
    if torch.any(torch.isnan(tensor)):
        raise ValueError("A " + tensor_description + " was found to have nan values.")


class BrainMaGeModel(PyTorchFLModel):

    def __init__(self, 
                 data, 
                 base_filters, 
                 min_learning_rate, 
                 max_learning_rate,
                 learning_rate_cycles_per_epoch,
                 loss_function,
                 validation_function, 
                 opt, 
                 device='cpu',
                 n_classes=4,
                 n_channels=4,
                 psize=[128,128,128],
                 smooth=1e-7,
                 use_penalties=False, 
                 validate_without_patches = False,
                 validate_with_fine_grained_dice = True, 
                 torch_threads=None, 
                 kmp_affinity=False, 
                 loss_function_kwargs={}, 
                 validation_function_kwargs={},
                 output_per_example_valscores=True,
                 val_input_shape = None,
                 val_output_shape = None,
                 **kwargs):
        super().__init__(data=data, device=device, **kwargs)

        if torch_threads is not None:
            torch.set_num_threads(torch_threads)
        if kmp_affinity:
            os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
                 
        self.device = device
        self.set_logger('openfl.model_and_data')

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
        self.which_validation = validation_function
        self.opt = opt
        #TODO: Binary classficition with one channel is currently not supported
        self.label_channels = self.n_classes
        self.base_filters = base_filters
        self.smooth = smooth
        self.which_model = self.__repr__()
        self.use_panalties = use_penalties

        self.loss_function_kwargs = loss_function_kwargs
        self.validation_function_kwargs = validation_function_kwargs

        # used only when using the gandlf_data object
        # (will we crop external zero-planes, infer, then pad output with zeros OR
        #  get outputs for multiple patches - fusing the outputs)
        self.validate_without_patches = validate_without_patches

        # Determines if we want our validation results to include separate values for whole-tumor, tumor-core, and
        # enhancing tumor, or to simply report the average of those
        self.validate_with_fine_grained_dice = validate_with_fine_grained_dice

        # Do we produce a list of validation scores (over samples of the val loader), or
        # do we output a single score resulting from the average of these?
        self.output_per_example_valscores = output_per_example_valscores

        # if not None, used to sanity check what input and output shapes are for validation pipeline 
        self.val_input_shape = val_input_shape
        self.val_output_shape = val_output_shape
        
        ############### CHOOSING THE LOSS AND VALIDATION FUNCTIONS ###################

        # hard coded for now
        #FIXME: Note dependency on this and loss_function_kwargs on total_valscore definition in validate method
        # I try to track this with self.validation_output_keys (below)
        if self.which_validation == 'brats_dice':
            self.validation_function = brats_dice
            if self.validate_with_fine_grained_dice:
                self.validation_output_keys = ['float_DICE_ET', 
                                               'float_DICE_TC', 
                                               'float_DICE_WT']
            else:
                self.validation_output_keys = ['float_DICE_AVG(ET,TC,WT)']
        elif self.which_validation == 'fets_phase2_validation':
            self.validation_function = fets_phase2_validation
            self.validation_output_keys = ['float_DICE_ET', 
                                           'float_DICE_TC', 
                                           'float_DICE_WT',
                                           'binary_DICE_ET', 
                                           'binary_DICE_TC', 
                                           'binary_DICE_WT', 
                                           'binary_Hausdorff95_ET', 
                                           'binary_Hausdorff95_TC', 
                                           'binary_Hausdorff95_WT', 
                                           'binary_Sensitivity_ET', 
                                           'binary_Sensitivity_TC', 
                                           'binary_Sensitivity_WT', 
                                           'binary_Specificity_ET', 
                                           'binary_Specificity_TC', 
                                           'binary_Specificity_WT']
        else:
            raise ValueError('The validation function {} is not currently supported'.format(self.which_validation))

        # old dc is now dice_loss_skipping_first_channel
        if self.which_loss == 'brats_dice_loss':
            self.loss_fn = brats_dice_loss
        elif self.which_loss == 'brats_dice_log_loss':
            self.loss_fn = brats_dice_log_loss
        elif self.which_loss == 'brats_dice_loss_w_background':
            self.loss_fn = brats_dice_loss_w_background
        elif self.which_loss == 'brats_dice_loss_w_crossentropy':
            self.loss_fn = brats_dice_loss_w_crossentropy
        elif self.which_loss == 'crossentropy':
            self.loss_fn = crossentropy
        elif self.which_loss == 'background_dice_loss':
            self.loss_fn = background_dice_loss
        elif self.which_loss == 'dice_loss_skipping_first_channel':
            self.loss_fn = dice_loss_skipping_first_channel
        elif self.which_loss == 'dice_loss_all_channels':
            self.loss_fn = dice_loss_all_channels
        elif self.which_loss == 'mirrored_brats_dice_loss':
            self.loss_fn = mirrored_brats_dice_loss
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

        # TODO: To sync learning rate cycle across collaborators, we assume each collaborator is training 
        # a set fraction of an epoch (rather than a set number of batches) otherwise use batch_num based cycle length
        cycle_length =  int((float(self.data.get_training_data_size())/float(self.data.batch_size)) / float(self.learning_rate_cycles_per_epoch))
        if cycle_length == 0:
            if self.data.get_training_data_size() == 0:
                cycle_length = 1
                self.logger.debug("No training data is present, so setting silly cyclic length for scheduler.")
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

    def sanity_check_val_input_shape(self, features):
        features_shape = list(features.shape)
        if (self.val_input_shape is not None) and (self.val_input_shape != features_shape):
            # FIXME: (replace with raised exception?)
            self.logger.debug('Features going into model during validation has shape {} when {} was expected.'.format(features_shape, self.val_input_shape))

    def sanity_check_val_output_shape(self, output):
        output_shape = list(output.shape)
        if (self.val_output_shape is not None) and (self.val_output_shape != output_shape):
            # FIXME: (replace with raised exception?)
            self.logger.debug('Output from the model during validation has shape {} when {} was expected.'.format(output_shape, self.val_output_shape))

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
            for i in range(0, self.n_classes):
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
            output = self(features)
            output = output.cpu()
        return output

    def train_batches(self, num_batches, use_tqdm=False):
        
        device = torch.device(self.device)

        ################################ LOGGING SOME STUFF ######################
        self.logger.debug("Hostname   : {}".format(str(os.getenv("HOSTNAME"))))
        sys.stdout.flush()

        self.logger.debug("Training batches: {}".format(len(self.data.train_loader.dataset)))
        sys.stdout.flush()

        self.logger.debug('Using device: {}'.format(device))
        if device.type == 'cuda':
            self.logger.debug('Memory Allocated: {}GB'.format(round(torch.cuda.memory_allocated(0)/1024**3, 1)))
            self.logger.debug('Memory Cached: {}GB'.format(round(torch.cuda.memory_cached(0)/1024**3, 1)))

        sys.stdout.flush()

        train_loader = self.data.get_train_loader()

        if train_loader == []:
            raise RuntimeError("Attempting to run training with an empty training loader.")

        if use_tqdm:
            train_loader = tqdm.tqdm(train_loader, desc="training for this round")

        total_loss = 0
        batch_num = 0
        
        # set to "training" mode
        self.train()
        while batch_num < num_batches:
                       
            for batch in train_loader:
                if batch_num >= num_batches:
                    break
                else:
                    # Load the batch and its ground truth
                    
                    # this is when we are using pt_brainmagedata
                    if ('features' in batch.keys()) and ('gt' in batch.keys()):
                        features = batch['features']
                        nan_check(tensor=features, tensor_description='features tensor')
                        mask = batch['gt']
                        nan_check(tensor=mask, tensor_description='ground truth mask tensor')
                    # this is when we are using gandlf loader   
                    else:
                        self.logger.debug("Training on batch with subjects: {}".format(batch['subject_id']))
                        features = torch.cat([batch[key][torchio.DATA] for key in self.channel_keys], dim=1).float()
                        nan_check(tensor=features, tensor_description='features tensor')
                        mask = batch['label'][torchio.DATA]
                        nan_check(tensor=mask, tensor_description='ground truth mask tensor')

                    mask = one_hot(mask, self.data.class_list).float()
                    nan_check(tensor=mask, tensor_description='one_hot ground truth mask tensor')
                        
                    # Loading features into device
                    features, mask = features.to(device), mask.to(device)
                    
                    # Making sure that the optimizer has been reset
                    self.optimizer.zero_grad()

                    # Forward Propagation to get the output from the models
                    output = self(features)
                    nan_check(tensor=output, tensor_description='model output tensor')
                    
                    # Computing the loss
                    loss = self.loss_fn(output=output, 
                                        target=mask, 
                                        num_classes=self.label_channels, 
                                        weights=self.dice_penalty_dict, 
                                        class_list=self.data.class_list, 
                                        to_scalar=False, 
                                        **self.loss_function_kwargs)
                    nan_check(tensor=loss, tensor_description='model loss tensor')

                    # Back Propagation for model to learn    
                    loss.backward()
                    #Updating the weight values
                    self.optimizer.step()
                    #Pushing the dice to the cpu and only taking its value
                    loss = loss.cpu().data.item()
                    total_loss += loss
                    self.lr_scheduler.step()

                    batch_num += 1

        # we return the average batch loss over all samples trained with this round
        # we also return the total batches used
        return {"loss": total_loss / num_batches, "num_batches_used": num_batches }

    def validate(self, use_tqdm=False, save_outputs=False, model_id=None, model_version=None, local_outputs_directory=None):
        
        if save_outputs:
            if (model_id is None) or (model_version is None) or (local_outputs_directory is None):
                raise ValueError('All of model_id, model_version, and local_outputs_directory need to be defined when using save_outputs.')
            outputs = []

        # dice results are dictionaries (keys provided by self.validation_output_keys)
        valscores = {key: [] for key in self.validation_output_keys}
        
        val_loader = self.data.get_val_loader()

        if val_loader == []:
            raise RuntimeError("Attempting to run validation with an empty val loader.")

        if use_tqdm:
            val_loader = tqdm.tqdm(val_loader, desc="validate")

        for subject in val_loader:
            # this is when we are using pt_brainmagedata
            if ('features' in subject.keys()) and ('gt' in subject.keys()):
                features = subject['features']
                nan_check(tensor=features, tensor_description='features tensor')
                mask = subject['gt']
                nan_check(tensor=mask, tensor_description='ground truth mask tensor')
        
                self.sanity_check_val_input_shape(features)
                output = self.infer_batch_with_no_numpy_conversion(features=features)
                nan_check(tensor=output, tensor_description='model output tensor')
                self.sanity_check_val_output_shape(output)
                    
            # using the gandlf loader   
            else:
                self.logger.debug("Validating with subject: {}".format(subject['subject_id']))
                features = torch.cat([subject[key][torchio.DATA] for key in self.channel_keys], dim=1).float()
                nan_check(tensor=features, tensor_description='features tensor')
                mask = subject['label'][torchio.DATA]
                nan_check(tensor=mask, tensor_description='ground truth mask tensor')

                if self.validate_without_patches:
                    self.sanity_check_val_input_shape(features)
                    output = self.data.infer_with_crop(model_inference_function=[self.infer_batch_with_no_numpy_conversion], 
                                                       features=features)
                    nan_check(tensor=output, tensor_description='model output tensor')
                    self.sanity_check_val_output_shape(output)
                else:
                    self.sanity_check_val_input_shape(features)
                    output = self.data.infer_with_crop_and_patches(model_inference_function=[self.infer_batch_with_no_numpy_conversion], 
                                                                   features=features)
                    nan_check(tensor=output, tensor_description='model output tensor')
                    self.sanity_check_val_output_shape(output)
                    
            if save_outputs:
                outputs.append(output.numpy())   
                
            # one-hot encoding of ground truth
            mask = one_hot(mask, self.data.class_list).float()
            nan_check(tensor=mask, tensor_description='one_hot ground truth mask tensor')
            
            # sanity check that the output and mask have the same shape
            if output.shape != mask.shape:
                raise ValueError('Model output and ground truth mask are not the same shape.')

            # FIXME: Create a more general losses.py module (with composability and aggregation)
            current_valscore = self.validation_function(output=output, 
                                                        target=mask, 
                                                        class_list=self.data.class_list, 
                                                        fine_grained=self.validate_with_fine_grained_dice, 
                                                        **self.validation_function_kwargs)
            for key, value in current_valscore.items():
                nan_check(tensor=torch.Tensor([value]), tensor_description='validation result with key {}'.format(key))

            # the dice results here are dictionaries (sum up the totals)
            for key in self.validation_output_keys:
                valscores[key].append(current_valscore[key])

        if save_outputs:
                if not os.path.exists(local_outputs_directory):
                    os.mkdir(local_outputs_directory)
                output_pardir = os.path.join(local_outputs_directory, model_id)
                if not os.path.exists(output_pardir):
                    os.mkdir(output_pardir)
                subdir_base = os.path.join(output_pardir, 'model_version_' + str(model_version) + '_output_for_validation_instance_')
                found_unused_subdir = False
                instance = -1
                subdirpath_to_use = None
                while (not found_unused_subdir) and (instance < 10):
                    instance += 1
                    subdirpath = subdir_base + str(instance)
                    if os.path.exists(subdirpath):
                        continue
                    else: 
                        os.mkdir(subdirpath)
                        subdirpath_to_use = subdirpath
                        found_unused_subdir = True
                if not found_unused_subdir:
                    raise ValueError('Already have 10 model output subdirs under {} for model {} and version {}.'.format(output_pardir, model_id, model_version))
                self.data.write_outputs(outputs=outputs, dirpath=subdirpath_to_use, class_list=self.data.class_list)

        if self.output_per_example_valscores:
            self.logger.debug("Producing per-example validation scores per key.")        
            return valscores
        else:
            self.logger.debug("Producing single float validation scores per key.")
            return {key: np.mean(scores_list) for key, scores_list in valscores.items()}



    

    

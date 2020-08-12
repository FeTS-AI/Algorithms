# The following code is modified from https://github.com/CBICA/BrainMaGe which has the following lisence:

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
import torch
import torch.optim as optim
from torch.autograd import Variable

from openfl import load_yaml
from models.pytorch import PyTorchFLModel
from .losses import MCD_loss, DCCE, CE, MCD_MSE_loss, dice_loss

# TODO: Run in CONTINUE_LOCAL or RESET optimizer modes for now, later ensure that the cyclic learning rate is properly handled for CONTINUE_GLOBAL.
# FIXME: do we really want to keep loss at 1-dice rather than -ln(dice)
# FIXME: Turn on data augmentation for training (double check all)

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
                 **kwargs):
        super().__init__(data=data, device=device, **kwargs)
        
        self.device = device

        # setting parameters as attributes
        # training parameters
        self.learning_rate = learning_rate
        self.which_loss = loss_function
        self.opt = opt
        # model parameters
        self.n_classes = self.data.n_classes
        self.binary_classification = self.n_classes == 2
        if self.binary_classification:
            self.label_channels = 1
        else:
            self.label_channels = self.n_classes
        self.base_filters = base_filters
        self.n_channels = self.data.n_channels
        self.which_model = self.__repr__()
        self.psize = self.data.psize 
        
        ############### CHOOSING THE LOSS FUNCTION ###################
        if self.which_loss == 'dc':
            self.loss_fn  = MCD_loss
        if self.which_loss == 'dcce':
            self.loss_fn  = DCCE
        if self.which_loss == 'ce':
            self.loss_fn = CE
        if self.which_loss == 'mse':
            self.loss_fn = MCD_MSE_loss
    
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

    def train_batches(self, num_batches, use_tqdm=False):

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
        if use_tqdm:
            train_loader = tqdm.tqdm(train_loader, desc="training for this round")

        total_round_training_loss = 0
        batch_num = 0

        # set to "training" mode
        self.train()
        while batch_num < num_batches:
                       
            total_loss = 0
            for subject in train_loader:
                if batch_num >= num_batches:
                    break
                else:
                    # Load the subject and its ground truth
                    features = subject['features']
                    mask = subject['gt']
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
                    loss = self.loss_fn(output.double(), mask.double(),num_class=self.label_channels)
                    # Back Propagation for model to learn
                    loss.backward()
                    #Updating the weight values
                    self.optimizer.step()
                    #Pushing the dice to the cpu and only taking its value
                    curr_loss = dice_loss(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()).cpu().data.item()
                    #train_loss_list.append(loss.cpu().data.item())
                    total_loss+=curr_loss
                    self.lr_scheduler.step()

                    # TODO: Not recommended? (https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/6)will try without
                    #torch.cuda.empty_cache()

                    batch_num += 1

            total_round_training_loss += total_loss
        # we return the average batch loss over all epochs trained this round
        return total_round_training_loss / num_batches

    def validate(self, use_tqdm=False):
        device = torch.device(self.device)       
        self.eval()
        
        total_loss = 0
        total_dice = 0
        
        val_loader = self.data.get_val_loader()
        if use_tqdm:
            val_loader = tqdm.tqdm(val_loader, desc="validate")

        for batch_idx, (subject) in enumerate(val_loader):
            with torch.no_grad():
                features = subject['features']
                mask = subject['gt']
                features, mask = features.to(device), mask.to(device)
                output = self(features.float())
                curr_loss = dice_loss(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()).cpu().data.item()
                total_loss+=curr_loss
                #Computing the dice score 
                curr_dice = 1 - curr_loss
                #Computing the total dice
                total_dice+= curr_dice
                #Computing the average dice
                average_dice = total_dice/(batch_idx + 1)

        return average_dice


    

    

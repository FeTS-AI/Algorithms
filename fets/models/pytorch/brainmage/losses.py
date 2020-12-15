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
import torch

def channel_dice_loss(output, target, smooth=1e-7):
    output = output.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (output * target).sum()
    return 1 - ((2. * intersection + smooth) / (output.sum() + target.sum() + smooth))



def channel_log_dice_loss(output, target, smooth=1e-7):
    output = output.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (output * target).sum()
    return -np.log((2. * intersection + smooth) / (output.sum() + target.sum() + smooth))


def ave_loss_over_channels(output, target, binary_classification, channel_loss_fn, **kwargs):
    if not binary_classification:
        # we will not count the background class (here in dim=0 of axis=1)
        output = output[:,1:,:,:]
        target = target[:,1:,:,:]
    total_dice = 0
    nb_nonbackground_classes = output.shape[1]
    for dim in range(nb_nonbackground_classes):
        output_channel = output[:,dim,:,:,:]
        target_channel = target[:,dim,:,:,:]
        total_dice += channel_loss_fn(output=output_channel, target=target_channel, **kwargs)
    return total_dice / nb_nonbackground_classes


def dice_loss(output, target, binary_classification, **kwargs):
    return ave_loss_over_channels(output=output, 
                                  target=target, 
                                  binary_classification=binary_classification, 
                                  channel_loss_fn=channel_dice_loss, 
                                  **kwargs)


def log_dice_loss(output, target, binary_classification, **kwargs):
    return ave_loss_over_channels(output=output, 
                                  target=target, 
                                  binary_classification=binary_classification, 
                                  channel_loss_fn=channel_log_dice_loss, 
                                  **kwargs)


def MCD_loss(pm, gt, num_class):
    acc_dice_loss = 0
    for i in range(0,num_class):
        acc_dice_loss += channel_dice_loss(gt[:,i,:,:,:],pm[:,i,:,:,:])
    acc_dice_loss/= num_class
    return acc_dice_loss

def MCD_loss_no_background(pm, gt, num_class):
    acc_dice_loss = 0
    for i in range(1,num_class):
        acc_dice_loss += channel_dice_loss(gt[:,i,:,:,:],pm[:,i,:,:,:])
    acc_dice_loss/= num_class
    return acc_dice_loss

# Setting up the Evaluation Metric
def dice(out, target):
    smooth = 1e-7
    oflat = out.view(-1)
    tflat = target.view(-1)
    intersection = (oflat * tflat).sum()
    return (2*intersection+smooth)/(oflat.sum()+tflat.sum()+smooth)


def CE(out,target):
    oflat = out.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    loss = torch.dot(-torch.log(oflat), tflat)/tflat.sum()
    return loss

def CCE(out, target, num_class):
    acc_ce_loss = 0
    for i in range(num_class):
        acc_ce_loss += CE(out[:,i,:,:,:],target[:,i,:,:,:])
    acc_ce_loss /= num_class
    return acc_ce_loss
        

def DCCE(out,target, n_classes):
    l = MCD_loss(out,target, n_classes) + CCE(out,target,n_classes)
    return l

def TV_loss(inp, target, alpha = 0.3, beta = 0.7):
    smooth = 1e-7
    iflat = inp.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - (intersection + smooth)/(alpha*iflat.sum() + beta*tflat.sum() + smooth)


def MCT_loss(inp, target, num_class):
    acc_tv_loss = 0
    for i in range(0,num_class):
        acc_tv_loss += TV_loss(inp[:,i,:,:,:],target[:,i,:,:,:])
    acc_tv_loss/= num_class
    return acc_tv_loss

def MSE(inp,target):
    iflat = inp.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    num = len(iflat)
    loss = (iflat - tflat)*(iflat - tflat)
    loss = loss.sum()
    loss = loss/num
    return loss
    
def MSE_loss(inp,target,num_classes):
    acc_mse_loss = 0
    for i in range(0,num_classes):
        acc_mse_loss += MSE(inp[:,i,:,:,:], target[:,i,:,:,:])
    acc_mse_loss/=num_classes
    return acc_mse_loss
    
def MCD_MSE_loss(inp,target,num_classes):
    l = MCD_loss(inp,target,num_classes) + 0.1*MSE_loss(inp,target,num_classes)
    return l




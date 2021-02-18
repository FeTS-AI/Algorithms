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


def clinical_dice(output, target, class_list, fine_grained=True, smooth=1e-7, **kwargs):
    # some sanity checks
    if output.shape != target.shape:
        raise ValueError('Shapes of output and target going into clinical_dice do not match.')
    if output.shape[1] != len(class_list):
        raise ValueError('The channel of output (and target) expected to enumerate class channels is not the right size.')

    fine_grained_results = clinical_dice_fine_grained(output=output, 
                                                      target=target, 
                                                      class_list=class_list, 
                                                      smooth=smooth,
                                                      **kwargs)

    if fine_grained:
        # here keys will be: 'ET', 'WT', and 'TC'
        return fine_grained_results    
    else:
        average = (fine_grained_results['ET'] + fine_grained_results['WT'] + fine_grained_results['TC']) / 3
        return {'AVG(ET,WT,TC)': average}


def clinical_dice_fine_grained(output, target, class_list, smooth=1e-7, **kwargs):
    # some sanity checks
    if output.shape != target.shape:
        raise ValueError('Shapes of output and target going into clinical_dice do not match.')
    if output.shape[1] != len(class_list):
        raise ValueError('The channel of output (and target) expected to enumerate class channels is not the right size.')

    # We detect two use_cases here, and force a change in the code when another is wanted.
    # In both cases, we rely on the order of class_list !!!
    if list(class_list) == [0, 1, 2, 4]:
        clinical_labels = False
    # In this case we track only enhancing tumor, whole tumor, and tumor core (no background class).
    elif list(class_list) == ['4', '1||2||4', '1||4']:
        clinical_labels = True
    else:
        raise ValueError('clinical dice is not yet designed for this model class_list: ', class_list)

    if clinical_labels:

        # we know it is channel 0 only because of know class_list above
        dice_for_enhancing = channel_dice(output=output[:,0,:,:,:], 
                                          target=target[:,0,:,:,:], 
                                          smooth=smooth, 
                                          **kwargs)
        # we know it is channel 1 only because of know class_list above
        dice_for_whole = channel_dice(output=output[:,1,:,:,:], 
                                      target=target[:,1,:,:,:], 
                                      smooth=smooth, 
                                      **kwargs)
        # we know it is channel 2 only because of know class_list above
        dice_for_core = channel_dice(output=output[:,2,:,:,:], 
                                     target=target[:,2,:,:,:], 
                                     smooth=smooth, 
                                     **kwargs)
    else:

        # enhancing_tumor ('4': channel 3 based on known class_list above)
        output_enhancing = output[:,3,:,:,:]
        target_enhancing = target[:,3,:,:,:]
        dice_for_enhancing = channel_dice(output=output_enhancing, 
                                          target=target_enhancing, 
                                          smooth=smooth, 
                                          **kwargs)
    
        # whole tumor ('1'|'2'|'4', ie channels 1, 2, or 3 based on known class_list above)
        output_whole = torch.max(output[:,1:,:,:,:],dim=1).values
        target_whole = torch.max(target[:,1:,:,:,:],dim=1).values
        dice_for_whole = channel_dice(output=output_whole, 
                                      target=target_whole, 
                                      smooth=smooth, 
                                      **kwargs)
    
        # tumor core ('1'|'4', ie channels 1 or 3 based on known class_list above)
        output_channels_1_3 = torch.cat([output[:,1,:,:,:], output[:,3,:,:,:]], dim=1)
        output_core = torch.max(output_channels_1_3,dim=1).values
        target_channels_1_3 = torch.cat([target[:,1,:,:,:], target[:,3,:,:,:]],dim=1)
        target_core = torch.max(target_channels_1_3,dim=1).values
        dice_for_core = channel_dice(output=output_core, 
                                     target=target_core, 
                                     smooth=smooth, 
                                     **kwargs)

    return {'ET': dice_for_enhancing, 'WT': dice_for_whole, 'TC': dice_for_core}


def clinical_dice_loss(output, target, class_list, smooth=1e-7, **kwargs):
    clin_dice = clinical_dice(output=output, 
                              target=target, 
                              class_list=class_list,
                              fine_grained=False, 
                              smooth=smooth, 
                              **kwargs)
    return 1 - clin_dice['AVG(ET,WT,TC)']


def clinical_dice_log_loss(output, target, class_list, smooth=1e-7, **kwargs):
    clin_dice = clinical_dice(output=output, 
                              target=target, 
                              class_list=class_list,
                              fine_grained=False, 
                              smooth=smooth, 
                              **kwargs)
                              
    if clin_dice['AVG(ET,WT,TC)'] <= 0:
        return 0
    else:
        return - torch.log(clin_dice['AVG(ET,WT,TC)'])


def channel_dice_loss(output, target, smooth=1e-7, **kwargs):
    return 1 - channel_dice(output=output, 
                            target=target, 
                            smooth=smooth, 
                            **kwargs)


def channel_log_dice_loss(output, target, smooth=1e-7, **kwargs):
    return -np.log(channel_dice(output=output, 
                                target=target, 
                                smooth=smooth, 
                                **kwargs))

def channel_dice(output, target, smooth=1e-7, to_scalar=False, **kwargs):
    output = output.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (output * target).sum()
    dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
    if to_scalar: 
        return dice.cpu().data.item()
    else:
        return dice 

def average_dice_over_channels(output, target, binary_classification, **kwargs):
    if not binary_classification:
        # we will not count the background class (here in dim=0 of axis=1)
        output = output[:,1:,:,:,:]
        target = target[:,1:,:,:,:]
    total_dice = 0
    nb_nonbackground_classes = output.shape[1]
    for dim in range(nb_nonbackground_classes):
        output_channel = output[:,dim,:,:,:]
        target_channel = target[:,dim,:,:,:]
        total_dice += channel_dice(output=output_channel, target=target_channel, **kwargs)
    return total_dice / nb_nonbackground_classes

def ave_loss_over_channels(output, target, binary_classification, channel_loss_fn, **kwargs):
    if not binary_classification:
        # we will not count the background class (here in dim=0 of axis=1)
        output = output[:,1:,:,:,:]
        target = target[:,1:,:,:,:]
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
    raise NotImplementedError('Find and fix this code')
    # return ave_loss_over_channels(output=output, 
    #                               target=target, 
    #                               binary_classification=binary_classification, 
    #                               channel_loss_fn=channel_log_dice_loss, 
    #                               **kwargs)


def MCD_loss(pm, gt, num_classes, weights = None, **kwargs):
    acc_dice_loss = 0
    for i in range(1,num_classes):
        current_dice_loss = channel_dice_loss(gt[:,i,:,:,:],pm[:,i,:,:,:])
        if weights is not None:
            current_dice_loss = current_dice_loss * weights[i]
        acc_dice_loss += current_dice_loss
    if weights is None:
        acc_dice_loss/= (num_classes-1)
    return acc_dice_loss


# Setting up the Evaluation Metric
def dice(out, target):
    smooth = 1e-7
    oflat = out.view(-1)
    tflat = target.view(-1)
    intersection = (oflat * tflat).sum()
    return (2*intersection+smooth)/(oflat.sum()+tflat.sum()+smooth)

# TODO: make the cross entropy w.r.t. something more like probabilities.
def CE(out,target, **kwargs):
    if bool(torch.sum(target) == 0): # contingency for empty mask
        return 0
    oflat = out.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    loss = torch.dot(-torch.log(oflat), tflat)/tflat.sum()
    return loss

def CCE(out, target, num_classes, **kwargs):
    acc_ce_loss = 0
    for i in range(num_classes):
        acc_ce_loss += CE(out[:,i,:,:,:],target[:,i,:,:,:], **kwargs)
    acc_ce_loss /= num_classes
    return acc_ce_loss
        

def DCCE(out,target, num_classes, **kwargs):
    l = MCD_loss(out,target, num_classes, **kwargs) + CCE(out,target,num_classes, **kwargs)
    return l

def TV_loss(inp, target, alpha = 0.3, beta = 0.7):
    smooth = 1e-7
    iflat = inp.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - (intersection + smooth)/(alpha*iflat.sum() + beta*tflat.sum() + smooth)


def MCT_loss(inp, target, num_classes):
    acc_tv_loss = 0
    for i in range(0,num_classes):
        acc_tv_loss += TV_loss(inp[:,i,:,:,:],target[:,i,:,:,:])
    acc_tv_loss/= num_classes
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



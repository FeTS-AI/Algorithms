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
from medpy.metric.binary import hd95

from fets.data.pytorch import new_labels_from_float_output
from GANDLF.utils import reverse_one_hot

######################################################
# some sanity checks to apply throughout this module #
######################################################

def check_is_binary_single(tensor):
    unique_values = torch.unique(tensor).numpy()
    if not set(unique_values).issubset(set([1.0, 0.0])):
        raise ValueError('The provided tensor is not binary as unique values are: {}'.format(unique_values))

def check_are_binary(output, target):
    binary_output = (set(torch.unique(output).numpy()).issubset(set([1.0, 0.0])))
    binary_target = (set(torch.unique(target).numpy()).issubset(set([1.0, 0.0])))
    if binary_output:
        if not binary_target:
            raise ValueError('The provided target is not binary.')
    else:
        if not binary_target:
            raise ValueError('Both the provided output and target are not binary.')
        else:
            raise ValueError('The provided output is not binary.')

def check_are_binary_numpy(output, target):
    binary_output = (set(np.unique(output)).issubset(set([1.0, 0.0])))
    binary_target = (set(np.unique(target)).issubset(set([1.0, 0.0])))
    if binary_output:
        if not binary_target:
            raise ValueError('The provided target is not binary.')
    else:
        if not binary_target:
            raise ValueError('Both the provided output and target are not binary.')
        else:
            raise ValueError('The provided output is not binary.')


def check_shapes_same(output, target):
    if output.shape != target.shape:
        raise ValueError('Shapes of output {} and target {} do not match.'.format(output.shape, target.shape))
    

def check_classes_enumerated_along_correct_axis(tensor, axis, num_classes):
    if tensor.shape[axis] != num_classes:
        raise ValueError('The idx={} channel of output (and target) should enumerate classes, but their shape is {} and there are {} classes.'.format(axis, tensor.shape, num_classes))


def check_axis_sum_is_one_single_tensor(tensor, dim):
    if not torch.all(torch.sum(tensor, dim=dim) - 1.0 < 1e-6):
        raise ValueError('The provided tensor of shape {} does not indicate softmax output along the provided axis {}.'.format(tensor.shape, dim))


def check_axis_sum_is_one(output, target, dim):
    sum_one_for_output = torch.all(torch.sum(output, dim=dim) - 1.0 < 1e-6)
    sum_one_for_target = torch.all(torch.sum(target, dim=dim) - 1.0 < 1e-6)
    if sum_one_for_output:
        if not sum_one_for_target:
            raise ValueError('The provided target does not indicate softmax output along the provided axis.')
    else:
        if not sum_one_for_target:
            raise ValueError('Both the provided output and target do not indicate softmax output along the provided axis.')
        else:
            raise ValueError('The provided output does not indicate softmax output along the provided axis.')


def check_values_in_open_interval(output, lower_limit, upper_limit):
    if torch.any(output <= lower_limit):
        raise ValueError('The provided output has voxel values lower than or equal to {}.'.format(lower_limit))
    if torch.any(output >= upper_limit):
        raise ValueError('The provided output has voxel values greater than or equal to {}.'.format(upper_limit))


######################################################
# some utilities to apply throughout this module #
######################################################

def int_output_from_softmax_output(output, class_list, class_axis=1):
    
    # infers class from float output by finding max of one-hot channels and applying class accordingly
    # requires a class list for which the class channel enumerates softmax output
    # for now this is only one class_list case
    assert class_list == [0, 1, 2, 4]

    check_classes_enumerated_along_correct_axis(tensor=output, axis=class_axis, num_classes=len(class_list))
    check_axis_sum_is_one_single_tensor(tensor=output, dim=class_axis)
    
    # (initialization only) get the indices for the correct class label for each voxel
    int_output = np.argmax(output, axis=class_axis)

    # replace indices with appropriate class label
    int_output.apply_(lambda idx : class_list[idx])

    

    return int_output


def apply_threshold(tensor, threshold=0.5):
    
    over_threshold = tensor >= threshold
    
    bin_tensor = torch.zeros_like(tensor)
    bin_tensor[over_threshold] = 1
    
    return bin_tensor


def binarize_output(output, class_list, modality, threshold=0.5, class_axis=1):
    if class_list == [0, 1, 2, 4]:
        # process the one_hot channels using argmax to get integer output
        integer_output = int_output_from_softmax_output(output=output, class_list=class_list, class_axis=class_axis)
        # convert [0, 1, 2, 4] to binary for modality
        binarized_output = torch.clone(integer_output)
    
        if modality == 'ET':
            binarized_output[binarized_output==1] = 0
            binarized_output[binarized_output==2] = 0
            binarized_output[binarized_output==4] = 1
        elif modality == 'TC':
            binarized_output[binarized_output==2] = 0
            binarized_output[binarized_output==4] = 1
        elif modality == 'WT':
            binarized_output[binarized_output==2] = 1
            binarized_output[binarized_output==4] = 1
        else:
            raise ValueError('Modality {} is not currently supported.'.format(modality))

    elif class_list == ['4', '1||4', '1||2||4']:
        check_classes_enumerated_along_correct_axis(tensor=output, axis=class_axis, num_classes=len(class_list))

        slices = [slice(None) for _ in output.shape]

        # select appropriate channel for modality, and convert floats to binary using threshold
        if modality == 'ET':
            slices[class_axis] = 0 
            binarized_output = apply_threshold(output[tuple(slices)], threshold=threshold)
        elif modality == 'TC':
            slices[class_axis] = 1
            binarized_output = apply_threshold(output[tuple(slices)], threshold=threshold)
        elif modality == 'WT':
            slices[class_axis] = 2
            binarized_output = apply_threshold(output[tuple(slices)], threshold=threshold)
        else:
            raise ValueError('Modality {} is not currently supported.'.format(modality))
          
    else:
        raise ValueError("Class list {} is not currently supported.".format(class_list))

    check_is_binary_single(binarized_output)
    
    return binarized_output



def brats_labels(output, target, class_list, binarized, **kwargs):
    # take output and target and create: (output_<task>, lable_<task>)
    # for tasks in ['enhancing', 'core', 'whole']
    # these can be binary (per-voxel) decisions (if binarized==True) or float valued
    
    if binarized:
        output_enhancing = binarize_output(output=output, 
                                           class_list=class_list, 
                                           modality='ET')
        
        output_core = binarize_output(output=output, 
                                      class_list=class_list, 
                                      modality='TC')
        
        output_whole = binarize_output(output=output, 
                                       class_list=class_list, 
                                       modality='WT')
       
    # We detect specific use_cases here, and force a change in the code when another is wanted.
    # In all cases, we rely on the order of class_list !!!
    if list(class_list) == [0, 1, 2, 4]:
        if not binarized:
            # signal is channel 3 based on known class_list
            output_enhancing = output[:,3,:,:,:]

            # core signal comes from channels 1 or 3 based on known class_list
            output_channels_1_3 = torch.cat([output[:,1:2,:,:,:], output[:,3:4,:,:,:]], dim=1)
            output_core = torch.max(output_channels_1_3,dim=1).values

            # whole signal comes from channels 1, 2, or 3 based on known class_list
            output_whole = torch.max(output[:,1:,:,:,:],dim=1).values
        
        # signal is channel 3 based on known class_list
        target_enhancing = target[:,3,:,:,:]

        # core signal comes from channels 1 or 3 based on known class_list
        target_channels_1_3 = torch.cat([target[:,1:2,:,:,:], target[:,3:4,:,:,:]],dim=1)
        target_core = torch.max(target_channels_1_3,dim=1).values
    
        # whole signal comes from channels 1, 2, or 3 based on known class_list
        target_whole = torch.max(target[:,1:,:,:,:],dim=1).values
    
    elif list(class_list) == ['4', '1||4', '1||2||4']:
        # In this case we track only enhancing tumor, tumor core, and whole tumor (no background class).
    
        if not binarized:

            # enhancing signal is channel 0 because of known class_list with fused labels
            output_enhancing = output[:,0,:,:,:]

            # core signal is channel 1 because of known class_list with fused labels
            output_core = output[:,1,:,:,:]

            # whole signal is channel 2 because of known class_list with fused labels
            output_whole = output[:,2,:,:,:]
        
        
        # enhancing signal is channel 0 because of known class_list with fused labels
        target_enhancing = target[:,0,:,:,:]
        
        # core signal is channel 1 because of known class_list with fused labels
        target_core = target[:,1,:,:,:]
        
        # whole signal is channel 2 because of known class_list with fused labels
        target_whole = target[:,2,:,:,:]
    else:
        raise ValueError('No implementation for this model class_list: ', class_list)

    check_shapes_same(output=output_enhancing, target=target_enhancing)
    check_shapes_same(output=output_core, target=target_core)
    check_shapes_same(output=output_whole, target=target_whole)

    return {'outputs': {'ET': output_enhancing, 
                        'TC': output_core,
                        'WT': output_whole},
            'targets': {'ET': target_enhancing, 
                        'TC': target_core, 
                        'WT': target_whole}}


######################################################
# now the validation and loss functions #
######################################################


def fets_phase2_validation(output, 
                           target, 
                           class_list, 
                           class_axis=1, 
                           to_scalar=True,
                           challenge_reduced_output=False,
                           challenge_remove_hausdorff=False, 
                           **kwargs):
    # some sanity checks
    check_shapes_same(output=output, target=target)
    check_classes_enumerated_along_correct_axis(tensor=output, axis=class_axis, num_classes=len(class_list))

    if not challenge_reduced_output:
        # get the binarized and non-binarized versions of the outputs and labels
        brats_val_data_non_binary = brats_labels(output=output, 
                                                target=target, 
                                                class_list=class_list, 
                                                binarized=False,
                                                **kwargs)
        outputs_non_binary = brats_val_data_non_binary['outputs']
        targets_non_binary = brats_val_data_non_binary['targets']


    brats_val_data_binary = brats_labels(output=output, 
                                         target=target, 
                                         class_list=class_list, 
                                         binarized=True,
                                         **kwargs)
    outputs_binary = brats_val_data_binary['outputs']
    targets_binary = brats_val_data_binary['targets']

    all_validation = {}

    if not challenge_reduced_output:
        # validation based on float outputs                                     
        all_validation.update(brats_dice(output=outputs_non_binary,
                                        target=targets_non_binary, 
                                        tag='float_', 
                                        data_already_processed=True,
                                        to_scalar=to_scalar, 
                                        **kwargs))
    
    # validation based on binarized outputs
    all_validation.update(brats_dice(output=outputs_binary,
                                     target=targets_binary, 
                                     tag='binary_',
                                     data_already_processed=True,
                                     to_scalar=to_scalar, 
                                     **kwargs))
    
    if not challenge_remove_hausdorff:
        all_validation.update(brats_hausdorff(output=outputs_binary,
                                            target=targets_binary, 
                                            tag='binary_', 
                                            data_already_processed=True,
                                            to_scalar=to_scalar, 
                                            **kwargs))

    if not challenge_reduced_output:
        all_validation.update(brats_sensitivity(output=outputs_binary,
                                                target=targets_binary, 
                                                tag='binary_', 
                                                data_already_processed=True,
                                                to_scalar=to_scalar, 
                                                **kwargs))

        all_validation.update(brats_specificity(output=outputs_binary,
                                                target=targets_binary, 
                                                tag='binary_', 
                                                data_already_processed=True,
                                                to_scalar=to_scalar, 
                                                **kwargs))

    return all_validation


def brats_sensitivity(output, 
                      target,
                      tag='', 
                      to_scalar=True, 
                      class_list=None,
                      data_already_processed=False, 
                      **kwargs):
    
    if not data_already_processed:
        if class_list is None:
            raise ValueError('class_list needs to be provided when data_already_processed is False.')
        # here we are being passed the raw output and target
        brats_val_data = brats_labels(output=output, 
                                    target=target, 
                                    class_list=class_list, 
                                    binarized=False,
                                    **kwargs)
        outputs = brats_val_data['outputs']
        targets = brats_val_data['targets']
        if tag != '':
            if tag != 'float_':
                raise ValueError('You are trying to tag float results with {} tag which is incorrect.'.format(tag))
        else:
            # overwriting default tag here
            tag = 'float_'
    else:
        outputs = output
        targets = target
    
    output_enhancing = outputs['ET'] 
    target_enhancing = targets['ET']

    output_core = outputs['TC'] 
    target_core = targets['TC'] 

    output_whole = outputs['WT'] 
    target_whole = targets['WT']

    sensitivity_for_enhancing = channel_sensitivity(output=output_enhancing, 
                                                    target=target_enhancing,
                                                    to_scalar=to_scalar, 
                                                    **kwargs)

    sensitivity_for_core = channel_sensitivity(output=output_core, 
                                               target=target_core,
                                               to_scalar=to_scalar, 
                                               **kwargs)

    sensitivity_for_whole = channel_sensitivity(output=output_whole, 
                                                target=target_whole,
                                                to_scalar=to_scalar, 
                                                **kwargs)

    return {tag + 'Sensitivity_ET': sensitivity_for_enhancing, 
            tag + 'Sensitivity_TC': sensitivity_for_core, 
            tag + 'Sensitivity_WT': sensitivity_for_whole}

def brats_specificity(output, 
                      target,
                      tag='',
                      to_scalar=True, 
                      class_list=None,
                      data_already_processed=False, **kwargs):
    
    if not data_already_processed:
        if class_list is None:
            raise ValueError('class_list needs to be provided when data_already_processed is False.')
        # here we are being passed the raw output and target
        brats_val_data = brats_labels(output=output, 
                                    target=target, 
                                    class_list=class_list, 
                                    binarized=False,
                                    **kwargs)
        outputs = brats_val_data['outputs']
        targets = brats_val_data['targets']
        if tag != '':
            if tag != 'float_':
                raise ValueError('You are trying to tag float results with {} tag which is incorrect.'.format(tag))
        else:
            # overwriting default tag here
            tag = 'float_'
    else:
        outputs = output
        targets = target
    
    output_enhancing = outputs['ET'] 
    target_enhancing = targets['ET']

    output_core = outputs['TC'] 
    target_core = targets['TC'] 

    output_whole = outputs['WT'] 
    target_whole = targets['WT']

    specificity_for_enhancing = channel_specificity(output=output_enhancing, 
                                                    target=target_enhancing,
                                                    to_scalar=to_scalar, 
                                                    **kwargs)

    specificity_for_core = channel_specificity(output=output_core, 
                                               target=target_core,
                                               to_scalar=to_scalar, 
                                               **kwargs)

    specificity_for_whole = channel_specificity(output=output_whole, 
                                                target=target_whole,
                                                to_scalar=to_scalar, 
                                                **kwargs)

    return {tag + 'Specificity_ET': specificity_for_enhancing, 
            tag + 'Specificity_TC': specificity_for_core, 
            tag + 'Specificity_WT': specificity_for_whole}


def brats_hausdorff(output, 
                    target, 
                    tag='', 
                    to_scalar=True, 
                    class_list=None, 
                    data_already_processed=False, 
                    **kwargs):

    if not data_already_processed:
        if class_list is None:
            raise ValueError('class_list needs to be provided when data_already_processed is False.')
        # here we are being passed the raw output and target
        brats_val_data = brats_labels(output=output, 
                                    target=target, 
                                    class_list=class_list,
                                    binarized=False, 
                                    **kwargs)
        outputs = brats_val_data['outputs']
        targets = brats_val_data['targets']
        if tag != '':
            if tag != 'float_':
                raise ValueError('You are trying to tag float results with {} tag which is incorrect.'.format(tag))
        else:
            # overwriting default tag here
            tag = 'float_'
    else:
        outputs = output
        targets = target

    output_enhancing = outputs['ET'] 
    target_enhancing = targets['ET']

    output_core = outputs['TC'] 
    target_core = targets['TC'] 

    output_whole = outputs['WT'] 
    target_whole = targets['WT']

    if to_scalar:
        # channel_hausdorff processes numpy arrays
        output_enhancing = output_enhancing.numpy().astype(np.int32) 
        target_enhancing = target_enhancing.numpy().astype(np.int32) 
        output_core  = output_core.numpy().astype(np.int32)   
        target_core = target_core.numpy().astype(np.int32) 
        output_whole  = output_whole.numpy().astype(np.int32)  
        target_whole = target_whole.numpy().astype(np.int32) 
    else:
        # I don't believe converting to and from numpy to utilize the channel_hausdorff function can be tracked in the graph 
        raise ValueError('Computing BraTS hausdorff for torch tensors in the compute graph is currently not supported.')
        
    

    hausdorff_for_enhancing = channel_hausdorff(output=output_enhancing, 
                                                target=target_enhancing)

    hausdorff_for_core = channel_hausdorff(output=output_core, 
                                           target=target_core)

    hausdorff_for_whole = channel_hausdorff(output=output_whole, 
                                            target=target_whole)

    return {tag + 'Hausdorff95_ET': hausdorff_for_enhancing, 
            tag + 'Hausdorff95_TC': hausdorff_for_core, 
            tag + 'Hausdorff95_WT': hausdorff_for_whole}


def brats_dice(output, 
               target, 
               fine_grained=True, 
               tag='', 
               smooth=1e-7, 
               class_list=None,
               class_axis=None, 
               data_already_processed=False, 
               **kwargs):
    
    if not data_already_processed:
        if class_list is None:
            raise ValueError('class_list needs to be provided when data_already_processed is False.')
        # here we are being passed the raw output and target
        brats_val_data = brats_labels(output=output, 
                                    target=target, 
                                    class_list=class_list,
                                    binarized=False, 
                                    **kwargs)
        outputs = brats_val_data['outputs']
        targets = brats_val_data['targets']
        if tag != '':
            if tag != 'float_':
                raise ValueError('You are trying to tag float results with {} tag which is incorrect.'.format(tag))
        else:
            # overwriting default tag here
            tag = 'float_'
    else:
        outputs = output
        targets = target
    fine_grained_results = brats_dice_fine_grained(outputs=outputs,
                                                   targets=targets, 
                                                   smooth=smooth, 
                                                   tag=tag,
                                                   **kwargs)
    if fine_grained:
        # here keys will be: 'ET', 'TC', and 'WT
        return fine_grained_results    
    else:
        average = (fine_grained_results[tag + 'DICE_ET'] + fine_grained_results[tag + 'DICE_TC'] + fine_grained_results[tag + 'DICE_WT']) / 3
        return {tag + 'DICE_AVG(ET,TC,WT)': average}


def brats_dice_fine_grained(outputs, targets, tag='', smooth=1e-7, **kwargs):

    output_enhancing = outputs['ET'] 
    target_enhancing = targets['ET']

    output_core = outputs['TC'] 
    target_core = targets['TC'] 

    output_whole = outputs['WT'] 
    target_whole = targets['WT']

    dice_for_enhancing = channel_dice(output=output_enhancing, 
                                      target=target_enhancing, 
                                      smooth=smooth, 
                                      **kwargs)
                                    
    dice_for_core = channel_dice(output=output_core, 
                                 target=target_core, 
                                 smooth=smooth, 
                                 **kwargs)

    dice_for_whole = channel_dice(output=output_whole, 
                                  target=target_whole, 
                                  smooth=smooth, 
                                  **kwargs)
    return {tag + 'DICE_ET': dice_for_enhancing, 
            tag + 'DICE_TC': dice_for_core, 
            tag + 'DICE_WT': dice_for_whole}

        
def brats_dice_loss(output, target, class_list, smooth=1e-7, **kwargs):

    b_dice = brats_dice(output=output, 
                        target=target, 
                        fine_grained=False, 
                        smooth=1e-7, 
                        class_list=class_list, 
                        data_already_processed=False, 
                        **kwargs)['float_DICE_AVG(ET,TC,WT)']
    
    return 1 - b_dice


def mirrored_brats_dice_loss(output, target, class_list, smooth=1e-7, **kwargs):
    return brats_dice_loss(output=output,
                           target=target, 
                           class_list=class_list, 
                           smooth=smooth, 
                           mirrored=True, 
                           **kwargs)
    

def brats_dice_log_loss(output, target, class_list, smooth=1e-7, **kwargs):
    
    b_dice = brats_dice(output=output, 
                        target=target, 
                        fine_grained=False, 
                        smooth=1e-7, 
                        class_list=class_list, 
                        data_already_processed=False, 
                        **kwargs)['float_DICE_AVG(ET,TC,WT)']

    if b_dice < 0:
        raise ValueError('BraTS dice should never be negative, something is wrong.')                          
    elif b_dice == 0:
        raise ValueError('BraTS dice should never be zero, something is wrong.')
    else:
        return - torch.log(b_dice)


def brats_dice_loss_w_crossentropy(output, 
                                   target, 
                                   class_list, 
                                   xent_weight=0.5, 
                                   smooth=1e-7, 
                                   **kwargs):
    brats_loss = brats_dice_loss(output=output, 
                                 target=target, 
                                 class_list=class_list, 
                                 smooth=smooth, 
                                 **kwargs)
    xent_loss = crossentropy(output=output, 
                             target=target, 
                             class_list=class_list, 
                             **kwargs)
    return brats_loss * (1 - xent_weight) + xent_loss * xent_weight


def brats_dice_loss_w_background(output, 
                                 target, 
                                 class_list, 
                                 background_weight=0.25, 
                                 smooth=1e-7, 
                                 **kwargs):
    if background_weight < 0 or background_weight > 1:
        raise ValueError('Background weight needs to between 0 an 1.')
    brats_loss = brats_dice_loss(output=output, 
                                 target=target, 
                                 class_list=class_list, 
                                 **kwargs)
    background_loss = background_dice_loss(output=output, 
                                           target=target, 
                                           class_list=class_list, 
                                           **kwargs)
    return brats_loss * (1-background_weight) + background_loss * background_weight


def background_dice_loss(output, target, class_list, smooth=1e-7, **kwargs):

    # some sanity checks
    check_shapes_same(output=output, target=target)
    check_classes_enumerated_along_correct_axis(tensor=output, axis=1, num_classes=len(class_list))

    # We detect two use_cases here, and force a change in the code when another is wanted.
    # In both cases, we rely on the order of class_list !!!
    if list(class_list) == [0, 1, 2, 4]:
        dice = channel_dice(output=output[:,0,:,:,:], 
                            target=target[:,0,:,:,:], 
                            smooth=smooth, 
                            **kwargs)
    # In this case background is identified via 1 - channel 2.
    elif list(class_list) == ['4', '1||4', '1||2||4']:

        dice = channel_dice(output=1-output[:,2,:,:,:], 
                            target=1-target[:,2,:,:,:], 
                            smooth=smooth, 
                            **kwargs)
    else:
        raise ValueError('No impelementation for this model class_list: ', class_list) 

    return 1 - dice


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


def channel_sensitivity(output, target, to_scalar=False, **kwargs):
    # compute TP/P 

    # the assumption is that output and target are binary
    check_are_binary(output=output, target=target)
    
    true_positives = torch.sum(torch.multiply(output, target))
    total_positives = torch.sum(target)

    if total_positives == 0:
        return 1.0
    else:
        score = true_positives / total_positives
        if to_scalar:
            score = score.item()
        return score


def channel_specificity(output, target, to_scalar=False, **kwargs):
    # compute TN/N

    # the assumption is that output and target are binary
    check_are_binary(output=output, target=target)

    true_negatives = torch.sum(torch.multiply(1 - output, 1 - target))
    total_negatives = torch.sum(1 - target)

    if total_negatives == 0:
        return 1.0
    else:
        score = true_negatives / total_negatives
        if to_scalar:
            score = score.item()
        return score


def channel_hausdorff(output, target):
    '''
    output, target are float values and contain 0s and 1s only
    '''
    check_are_binary_numpy(output=output, target=target)
    # check for either array having all zeros
    output_zeros = np.sum(output) == 0.0
    target_zeros = np.sum(target) == 0.0

    # if both arrays are all zero, return 0
    if output_zeros and target_zeros:
        return 0
    # if exactly one is all zeros, return max distance across volume (distance between corners)
    elif output_zeros or target_zeros:
        return np.linalg.norm([entry -1 for entry in output.shape])
    else:
        return hd95(output, target)

def channel_dice(output, target, smooth=1e-7, to_scalar=False, mirrored=False, **kwargs):
    # this dice is appropriate for  a single channel
    # examples are: background only, whole tumor only, enhancing tumor only, tumor core only
    def straight_dice(output, target, smooth, to_scalar, **kwargs):
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (output * target).sum()
        dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
        if to_scalar: 
            return dice.cpu().data.item()
        else:
            return dice 
    dice = straight_dice(output=output,
                         target=target, 
                         smooth=smooth, 
                         to_scalar=to_scalar, 
                         **kwargs)
    if not mirrored:
        return dice
    else:
        dice_on_mirror = straight_dice(output=1 - output,
                                       target=1 - target, 
                                       smooth=smooth, 
                                       to_scalar=to_scalar, 
                                       **kwargs)
        return (dice + dice_on_mirror)/2


def ave_loss_over_channels(output, target, channels, channel_loss_fn, channels_dim=1, **kwargs):

    # sanity check
    check_shapes_same(output=output, target=target)
    
    total_dice = 0
    nb_classes = output.shape[channels_dim]
    if np.amin(channels) < 0 or np.amax(channels) >= nb_classes:
        raise ValueError('Provided channels: {} are not consistent with the output (target) shape: {} and channels_dim of {}'.format(channels, output.shape, channels_dim))
    slices = [slice(None) for _ in output.shape] 
    for idx in channels:
        slices[channels_dim] = idx
        output_channel = output[tuple(slices)]
        target_channel = target[tuple(slices)]
        total_dice += channel_loss_fn(output=output_channel, target=target_channel, **kwargs)
    return total_dice / len(channels)


def dice_loss(output, target, skip=[], channels_dim=1, **kwargs):
    # skip is a list of channels to skip in the average

    # sanity check
    check_shapes_same(output=output, target=target)

    for idx in skip:
        if idx < 0 or idx >= output.shape[channels_dim]:
            raise ValueError('Skip channel out of range. Found skip idx: {} when channels_dim is {} and output shape is {}'.format(idx, channels_dim, output.shape))
    channels = []
    for idx in range(output.shape[channels_dim]):
        if idx not in skip:
            channels.append(idx)

    return ave_loss_over_channels(output=output, 
                                  target=target, 
                                  channels=channels, 
                                  channel_loss_fn=channel_dice_loss,
                                  channels_dim=channels_dim, 
                                  **kwargs)


def dice_loss_skipping_first_channel(output, target, **kwargs):
    return dice_loss(output=output, target=target, skip=[0], **kwargs)


def dice_loss_all_channels(output, target, **kwargs):
    return dice_loss(output=output, target=target, **kwargs)


def channel_binary_crossentropy(output, target, **kwargs):
    # computes the average over pixels of binary cross entropy for a single channel output 
    # each component in output should be a confidence (in (0,1) of the 1 outcome (other outcome being 0)

    # sanity check
    check_shapes_same(output=output, target=target)
    check_values_in_open_interval(output=output, lower_limit=0.0, upper_limit=1.0)
    
    output = torch.flatten(output)
    target = torch.flatten(target)
    pixel_xent_sum = -torch.dot(torch.log(output), target) - torch.dot(torch.log(1-output), (1-target))
    return pixel_xent_sum / (output.size().numel())
    

def crossentropy_over_channels(output, target, class_list, axis, **kwargs):
    # computes the average over pixels of cross entropy for multi-class classification 
    # for each pixel the provided axis should enumerate a multi-class confidence vector
    # (so sum over this axis should give 1 for every pixel, and no pixel should ever be exactly zero)

    # sanity checks
    check_shapes_same(output=output, target=target)
    check_classes_enumerated_along_correct_axis(tensor=output, axis=axis, num_classes=len(class_list))
    check_axis_sum_is_one(output=output, target=target, dim=axis)
    check_values_in_open_interval(output=output, lower_limit=0.0, upper_limit=1.0)
    

    # initialization
    slices = [slice(None) for _ in output.shape]
    pixel_xent_sum = 0

    for idx in range(output.shape[axis]):
        slices[axis] = idx

        partial_output = torch.flatten(output[tuple(slices)])
        partial_target = torch.flatten(target[tuple(slices)])
        pixel_xent_sum += -torch.dot(torch.log(partial_output), partial_target)
        
    return pixel_xent_sum / (output[tuple(slices)].size().numel())


def crossentropy(output, target, class_list, **kwargs):
    #FIXME: Do we really want to by default include the background channel in the first class_list case?
    #       Or should we at least make that configurable?

    class_axis = 1

    # sanity checks
    check_shapes_same(output=output, target=target)
    check_classes_enumerated_along_correct_axis(tensor=output, axis=class_axis, num_classes=len(class_list))

    # We detect two use_cases here, and force a change in the code when another is wanted.
    # In both cases, we rely on the order of class_list !!!
    if list(class_list) == [0, 1, 2, 4]:
        return crossentropy_over_channels(output=output, 
                                          target=target, 
                                          class_list=class_list, 
                                          axis=1, 
                                          **kwargs)
    elif list(class_list) == ['4', '1||4', '1||2||4']:
        # here we have a cross-entropy associated to each task (classes relate to independent tasks)
        xent_sum_across_tasks = 0
        for channel in range(output.shape[class_axis]):
           slices = [slice(None) for _ in output.shape]
           slices[class_axis] = channel 
           xent_sum_across_tasks += channel_binary_crossentropy(output=output[slices], 
                                                                target=target[slices], 
                                                                **kwargs)
        return xent_sum_across_tasks / len(class_list)
    else:
        raise ValueError('No implementation for this model class_list: ', class_list) 


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



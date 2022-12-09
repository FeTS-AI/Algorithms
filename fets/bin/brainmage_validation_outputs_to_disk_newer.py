import argparse
import os
import shutil
import numpy as np
import pickle as pkl

import SimpleITK as sitk
import torch
import torchio

from openfl import split_tensor_dict_for_holdouts, hash_string
from openfl.proto.protoutils import load_legacy_model_protobuf, load_proto, tensor_proto_to_numpy_array
from openfl.proto.collaborator_aggregator_interface_pb2 import TensorProto, ExtraModelInfo
from openfl.flplan import create_data_object_with_explicit_data_path, parse_fl_plan, create_model_object

from fets.data.pytorch.gandlf_data import GANDLFData
from fets.models.pytorch.brainmage.brainmage import BrainMaGeModel
from fets.models.pytorch.brainmage.losses import fets_phase2_validation

from GANDLF.utils import one_hot

##################################################################################
# The functions below are copied over from: 
# (Brandon Edwards original author) https://github.com/FETS-AI/Challenge/blob/main/Task_1/fets_challenge/inference.py
def nan_check(tensor, tensor_description):
    tensor = tensor.cpu()
    if torch.any(torch.isnan(tensor)):
        raise ValueError("A " + tensor_description + " was found to have nan values.")


def binarize(array, threshold=0.5):
    """
    Get binarized output using threshold. 
    """
    
    if (np.amax(array) > 1.0) or (np.amin(array) < 0.0):
        raise ValueError('Voxel value fed to lambda in converting to original labels was out of range.')
    
    # obtain binarized output
    binarized = array.copy()
    zero_mask = (binarized <= threshold)
    binarized[zero_mask] = 0.0
    binarized[~zero_mask] = 1.0
    
    return binarized


def get_binarized_and_belief(array, threshold=0.5):
    
    """
    Get binarized output using threshold and report the belief values for each of the
    channels along the last axis. Belief value is a list of indices along the last axis, and is 
    determined by the order of how close (closest first) each binarization is from its original in this last axis
    so belief should have the same shape as arrray.
    Assumptions:
    - array is float valued in the range [0.0, 1.0]
    """
    
    # check assumption above
    if (np.amax(array) > 1.0) or (np.amin(array) < 0.0):
        raise ValueError('Voxel value fed to lambda in converting to original labels was out of range.')
        
    # obtain binarized output
    binarized = binarize(array=array, threshold=threshold)
    
    # we will sort from least to greatest, so least suspicion is what we will believe
    raw_suspicion = np.absolute(array - binarized)
    
    belief = np.argsort(raw_suspicion, axis=-1)
    
    return binarized, belief

def replace_initializations(done_replacing, array, mask, replacement_value, initialization_value):
    """
    Replace in array[mask] intitialization values with replacement value, 
    ensuring that the locations to replace all held initialization values
    """
    
    # sanity check
    if np.any(mask) and done_replacing:
        raise ValueError('Being given locations to replace and yet told that we are done replacing.')
        
    # check that the mask and array have the same shape
    if array.shape != mask.shape:
        raise ValueError('Attempting to replace using a mask shape: {} not equal to the array shape: {}'.format(mask.shape, array.shape))
    
    # check that the mask only points to locations with initialized values
    if np.any(array[mask] != initialization_value):
        raise ValueError('Attempting to overwrite a non-initialization value.')
        
    array[mask] = replacement_value
    
    done_replacing = np.all(array!=initialization_value)
    
    return array, done_replacing


def check_subarray(array1, array2):
    """
    Checks to see where array2 is a subarray of array1.
    Assumptions:
    - array2 has one axis and is equal in length to the last axis of array1
    """
    
    # check assumption
    if (len(array2.shape) != 1) or (array2.shape[0] != array1.shape[-1]):
        print(f'Shapes of array1 and array2 are: {array1.shape}, {array2.shape}')
        raise ValueError('Attempting to check for subarray equality when shape assumption does not hold.')
        
    return np.all(array1==array2, axis=-1)



def convert_to_original_labels(array, threshold=0.5, initialization_value=999):
    """
    array has float output in the range [0.0, 1.0]. 
    Last three channels are expected to correspond to ET, TC, and WT respecively.
    
    """
    
    binarized, belief = get_binarized_and_belief(array=array, threshold=threshold)
    
    #sanity check
    if binarized.shape != belief.shape:
        raise ValueError('Sanity check did not pass.')
        
    # initialize with a crazy label we will be sure is gone in the end
    slice_all_but_last_channel = tuple([slice(None) for _ in array.shape[:-1]] + [0])
    original_labels = initialization_value * np.ones_like(array[slice_all_but_last_channel])
    
    # the outer keys correspond to the binarized values
    # the inner keys correspond to the order of indices comingn from argsort(ascending) on suspicion, i.e. 
    # how far the binarized sigmoid outputs were from the original sigmoid outputs 
    #     for example, (2, 1, 0) means the suspicion from least to greatest was: 'WT', 'TC', 'ET'
    #     (recall that the order of the last three channels is expected to be: 'ET', 'TC', and 'WT')
    mapper = {(0, 0, 0): 0, 
              (1, 1, 1): 4,
              (0, 1, 1): 1,
              (0, 0, 1): 2,
              (0, 1, 0): {(2, 0, 1): 0,
                          (2, 1, 0): 0, 
                          (1, 0, 2): 1,
                          (1, 2, 0): 1,
                          (0, 2, 1): 0,
                          (0, 1, 2): 1}, 
              (1, 1, 0): {(2, 0, 1): 0,
                          (2, 1, 0): 0, 
                          (1, 0, 2): 4,
                          (1, 2, 0): 4,
                          (0, 2, 1): 4,
                          (0, 1, 2): 4},
              (1, 0, 1): {(2, 0, 1): 4,
                          (2, 1, 0): 2, 
                          (1, 0, 2): 2,
                          (1, 2, 0): 2,
                          (0, 2, 1): 4,
                          (0, 1, 2): 4}, 
              (1, 0, 0): {(2, 0, 1): 0,
                          (2, 1, 0): 0, 
                          (1, 0, 2): 0,
                          (1, 2, 0): 0,
                          (0, 2, 1): 4,
                          (0, 1, 2): 4}}
    
    
    
    done_replacing = False
    
    for binary_key, inner in mapper.items():
        mask1 = check_subarray(array1=binarized, array2=np.array(binary_key))
        if isinstance(inner, int):
            original_labels, done_replacing = replace_initializations(done_replacing=done_replacing, 
                                                                      array=original_labels, 
                                                                      mask=mask1, 
                                                                      replacement_value=inner, 
                                                                      initialization_value=initialization_value)
        else:
            for inner_key, inner_value in inner.items():
                mask2 = np.logical_and(mask1, check_subarray(array1=belief, array2=np.array(inner_key)))
                original_labels, done_replacing = replace_initializations(done_replacing=done_replacing,
                                                                          array=original_labels, 
                                                                          mask=mask2, 
                                                                          replacement_value=inner_value, 
                                                                          initialization_value=initialization_value)
        
    if not done_replacing:
        raise ValueError('About to return so should have been done replacing but told otherwise.')
        
    return original_labels.astype(np.uint8)

##################################################################################




################################################################
# Hard coded parameters (Make sure these apply for your model) #
################################################################

# hard-coded class_list
class_list = ['4', '1||4', '1||2||4']
val_input_shape = [1, 4, 240, 240, 155]
val_output_shape = [1, len(class_list), 240, 240, 155]
        


def subject_to_feature_and_label(subject, class_list):
    # get
    features = torch.cat([subject[key][torchio.DATA] for key in ['1', '2', '3', '4']], dim=1)
    nan_check(tensor=features, tensor_description='features tensor')   
    print("Constructed features from subject with shape", features.shape)

    if subject['label'] != ['NA']:
        label = subject['label'][torchio.DATA]
    else:
        label = subject['1'][torchio.DATA]
    # one-hot encoding of ground truth
    label = one_hot(label, class_list).float()
    nan_check(tensor=label, tensor_description='one_hot ground truth label tensor')
    print("Constructed label from subject with shape",label.shape)
        
    return features, label


def load_model(directory):
    extra_model_info = load_proto(os.path.join(directory, 'ExtraModelInfo.pbuf'), proto_type=ExtraModelInfo)

    tensor_dict_from_proto = {}
    for t in extra_model_info.tensor_names:
        t_hash = hash_string(t)
        tensor_proto = load_proto(os.path.join(directory, '{}.pbuf'.format(t_hash)), proto_type=TensorProto)
        if t != tensor_proto.name:
            raise RuntimeError("Loaded the wrong tensor! Meant to load: {} did load: {} read file: {}".format(t, t.name, t_hash))
        tensor_dict_from_proto[t] = tensor_proto_to_numpy_array(tensor_proto)

    return tensor_dict_from_proto


def nan_check(tensor, tensor_description='model output'):
    tensor = tensor.cpu()
    if torch.any(torch.isnan(tensor)):
        raise ValueError("A " + tensor_description + " was found to have nan values.")


def sanity_check_val_input_shape(features, val_input_shape):
    features_shape = list(features.shape)
    if features_shape != val_input_shape:
        raise ValueError('Features going into model during validation have shape {} when {} was expected.'.format(features_shape, val_input_shape))


def sanity_check_val_output_shape(output, val_output_shape):
    output_shape = list(output.shape)
    if output_shape != val_output_shape:
        raise ValueError('Output from the model during validation has shape {} when {} was expected.'.format(output_shape, val_output_shape))


#########################################################################################
# Main will require a virtual environment with Algorithms, GANDLF, and OpenFL installed #
#########################################################################################
def main(data_path, 
         plan_path,
         model_weights_path_wt,
         model_weights_path_et,
         model_weights_path_tc, 
         output_pardir, 
         model_output_tag,
         device,
         process_training_data=False):

    flplan = parse_fl_plan(plan_path)

    channel_to_region = {
        0: 'ET',
        1: 'TC',
        2: 'WT'
    }

    # make sure the class list we are using is compatible with the hard-coded class_list above
    if flplan['data_object_init']['init_kwargs']['class_list'] != class_list:
        raise ValueError('We currently only support class_list=', class_list)

    if process_training_data:
        # patch the plan to set the data use to inference
        flplan['data_object_init']['init_kwargs']['data_usage'] = 'scoring'    

    # construct the data object
    data = create_data_object_with_explicit_data_path(flplan=flplan, data_path=data_path, data_usage='inference')

    # code is written with assumption we are using the gandlf data object
    if not issubclass(data.__class__, GANDLFData):
        raise ValueError('This script is currently assumed to be using a child of fets.data.pytorch.gandlf_data.GANDLFData, you are using: ', data.__class__.__name__)

    # construct the model object (requires cpu since we're passing [padded] whole brains)
    model = create_model_object(flplan=flplan, data_object=data, model_device=device)

    # code is written with assumption we are using the brainmage object
    if not issubclass(model.__class__, BrainMaGeModel):
        raise ValueError('This script is currently assumed to be using a child of fets.models.pytorch.brainmage.BrainMaGeModel, you are using: ', data.__class__.__name__)

    # get the holdout tensors
    _, holdout_tensors = split_tensor_dict_for_holdouts(None, model.get_tensor_dict())

    if not os.path.exists(output_pardir):
        os.mkdir(output_pardir)

    subdir_to_score = {}
    score_outpath = os.path.join(output_pardir, model_output_tag + '_subdirs_to_scores.pkl')

    # we either want the validation loader or the inference loader
    if process_training_data:
        # inference loader doesn't split the data, so the training data will get processed as well
        loader = data.get_scoring_loader()
    else:
        # otherwise, we only want the validation loader
        loader = data.get_inference_loader()#get_val_loader()

    print("\nWill be scoring {} samples.\n".format(len(loader)))

    # make all paths canonical
    model_weights_path_wt = os.path.realpath(model_weights_path_wt)
    model_weights_path_et = os.path.realpath(model_weights_path_et)
    model_weights_path_tc = os.path.realpath(model_weights_path_tc)

    # determine unique models
    unique_model_paths = set([model_weights_path_wt, model_weights_path_et, model_weights_path_tc])
    print('loading models:')
    for p in unique_model_paths:
        print('\t', p)

    # load the unique models
    model_path_to_weights = {p: {**load_model(p), **holdout_tensors} for p in unique_model_paths}

    # map unique models to channels
    model_path_to_channels = {p: [] for p in unique_model_paths}
    for p in unique_model_paths:
        if p == model_weights_path_wt:
            model_path_to_channels[p].append(2)
        if p == model_weights_path_tc:
            model_path_to_channels[p].append(1)
        if p == model_weights_path_et:
            model_path_to_channels[p].append(0)

    print(model_path_to_channels)

    for subject in loader:
        # infer the subject name from the label path
        key_to_consider = '1'
        label_present = False
        if subject['label'] != ['NA']:
            key_to_consider = 'label'
            label_present = True
        label_path = subject[key_to_consider]['path'][0]
        label_file = label_path.split('/')[-1]
        subdir_name = label_path.split('/')[-2]

        first_mode_path = subject['1']['path'][0]

        #prep the path for the output files
        output_subdir = os.path.join(output_pardir, subdir_name)
        if not os.path.exists(output_subdir):
            os.mkdir(output_subdir)
        inference_outpath = os.path.join(output_subdir, subdir_name + model_output_tag + '_seg.nii.gz')
       
        # copy the label file over to the output subdir
        copy_label_path = os.path.join(output_subdir, label_file)
        shutil.copyfile(label_path, copy_label_path)
        
        features, ground_truth = subject_to_feature_and_label(subject=subject, class_list=class_list)

        
        
        # skip samples of the wrong shape
        try:
            sanity_check_val_input_shape(features=features, val_input_shape=val_input_shape)
        except ValueError as e:
            print("Sanity check for", subdir_name, "failed with exception:")
            print(getattr(e, 'message', repr(e)))
            print("skipping subject")
            continue

        # Infer with patching
        output = None
        for path, weights in model_path_to_weights.items():
            print('Running inference with', path)
            # have to copy due to pop :(
            model.set_tensor_dict(weights.copy(), with_opt_vars=False)
            o = model.data.infer_with_crop_and_patches(model_inference_function=[model.infer_batch_with_no_numpy_conversion], features=features)
            if output is None:
                # the first model sets the base output
                output = o
            else:
                # determine which region(s) this model is used for
                for channel in model_path_to_channels[path]:
                    output[:, channel] = o[:, channel]

                    # log update of channel
                    print('Used model', path, 'to update channel', channel, '({})'.format(channel_to_region[channel]))

        nan_check(tensor=output)
        nan_check(tensor=output, tensor_description='model output tensor')
        sanity_check_val_output_shape(output=output, val_output_shape=val_output_shape)

        # get the validation scores
        dice_dict = fets_phase2_validation(output=output, 
                                        target=ground_truth, 
                                        class_list=class_list, 
                                        to_scalar=True)

        if subdir_to_score.get(subdir_name) is not None:
            raise ValueError('Trying to overwrite a second score for the subidir: {}'.format(subdir_name))
        subdir_to_score[subdir_name] = dice_dict

        if label_present:
            print("\nScores for record {} were: {}\n".format(subdir_name, dice_dict))


        output = np.squeeze(output.cpu().numpy())
        output_shape_length = len(output.shape)
        output = np.transpose(output, [idx+1 for idx in range(output_shape_length-1)]+[0])
        print("Just changed output to new shape: ", output.shape)

        # convert to class_list of three channels ET, TC, WT
        # transpose here to account for differences with original  
        four_channel_output = np.transpose(convert_to_original_labels(array=output), [2, 1, 0])


        # convert array to SimpleITK image 
        image = sitk.GetImageFromArray(four_channel_output)

        image.CopyInformation(sitk.ReadImage(first_mode_path))

        print("\nWriting inference NIfTI image of shape {} to {}".format(four_channel_output.shape, inference_outpath))
        sitk.WriteImage(image, inference_outpath)
        if label_present:
            print("\nCorresponding DICE scores were: ")
            print("{}\n\n".format(dice_dict))


    ## commenting this because of the case when ground truth files are absent, we still get _some_ information for debugging
    # print("Saving subdir_name_to_scores at: ", score_outpath)
    with open(score_outpath, 'wb') as _file:
        pkl.dump(subdir_to_score, _file)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-dp', type=str, required=True, help='Absolute path to the data folder.')
    parser.add_argument('--plan_path', '-pp', type=str, required=True, help='Absolute path to the plan file.')
    parser.add_argument('--model_weights_path_wt', '-WT', type=str, required=True)
    parser.add_argument('--model_weights_path_et', '-ET', type=str, required=True)
    parser.add_argument('--model_weights_path_tc', '-TC', type=str, required=True)
    parser.add_argument('--output_pardir', '-op', type=str, required=True)
    parser.add_argument('--model_output_tag', '-mot', type=str, default='test_tag')
    # parser.add_argument('--legacy_model_flag', '-lm', action='store_true')
    parser.add_argument('--device', '-dev', type=str, default='cpu', required=False)
    parser.add_argument('--process_training_data', '-ptd', action='store_true')
    args = parser.parse_args()
    main(**vars(args))

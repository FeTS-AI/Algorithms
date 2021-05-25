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

    label = subject['label'][torchio.DATA]
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
         model_weights_path, 
         output_pardir, 
         model_output_tag,
         device, 
         legacy_model_flag=False):

    flplan = parse_fl_plan(plan_path)

    # make sure the class list we are using is compatible with the hard-coded class_list above
    if flplan['data_object_init']['init_kwargs']['class_list'] != class_list:
        raise ValueError('We currently only support class_list=', class_list)

    # construct the data object
    data = create_data_object_with_explicit_data_path(flplan=flplan, data_path=data_path)

    # code is written with assumption we are using the gandlf data object
    if not issubclass(data.__class__, GANDLFData):
        raise ValueError('This script is currently assumed to be using a child of fets.data.pytorch.gandlf_data.GANDLFData, you are using: ', data.__class__.__name__)

    # construct the model object (requires cpu since we're passing [padded] whole brains)
    model = create_model_object(flplan=flplan, data_object=data, model_device=device)

    # code is written with assumption we are using the brainmage object
    if not issubclass(model.__class__, BrainMaGeModel):
        raise ValueError('This script is currently assumed to be using a child of fets.models.pytorch.brainmage.BrainMaGeModel, you are using: ', data.__class__.__name__)

    # legacy models are defined in a single file, newer ones have a folder that holds per-layer files
    if legacy_model_flag:
        tensor_dict_from_proto = load_legacy_model_protobuf(model_weights_path) 
    else:
        tensor_dict_from_proto = load_model(model_weights_path)

    # restore any tensors held out from the proto
    _, holdout_tensors = split_tensor_dict_for_holdouts(None, model.get_tensor_dict())        
    tensor_dict = {**tensor_dict_from_proto, **holdout_tensors}
    model.set_tensor_dict(tensor_dict, with_opt_vars=False) 

    print("\nWill be running inference on {} validation samples.\n".format(model.get_validation_data_size()))

    if not os.path.exists(output_pardir):
        os.mkdir(output_pardir)

    subdir_to_score = {}
    score_outpath = os.path.join(output_pardir, model_output_tag + '_subdirs_to_scores.pkl')

    for subject in data.get_val_loader():
        
        # infer the subject name from the label path
        label_path = subject['label']['path'][0]
        subdir_name = label_path.split('/')[-2]
        
        features, ground_truth = subject_to_feature_and_label(subject=subject, class_list=class_list)
                                    
        # Infer with patching
        sanity_check_val_input_shape(features=features, val_input_shape=val_input_shape)
        output = model.data.infer_with_crop_and_patches(model_inference_function=[model.infer_batch_with_no_numpy_conversion], 
                                                        features=features)
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

        print("\nScores for record {} were: {}\n".format(subdir_name, dice_dict))
        
    print("Saving subdir_name_to_scores at: ", score_outpath)
    with open(score_outpath, 'wb') as _file:
        pkl.dump(subdir_to_score, _file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-dp', type=str, required=True, help='Absolute path to the data folder.')
    parser.add_argument('--plan_path', '-pp', type=str, required=True, help='Absolute path to the plan file.')
    parser.add_argument('--model_weights_path', '-mwp', type=str, required=True)
    parser.add_argument('--output_pardir', '-op', type=str, required=True)
    parser.add_argument('--model_output_tag', '-mot', type=str, default='test_tag')
    parser.add_argument('--legacy_model_flag', '-lm', action='store_true')
    parser.add_argument('--device', '-dev', type=str, default='cpu', required=False)
    args = parser.parse_args()
    main(**vars(args))

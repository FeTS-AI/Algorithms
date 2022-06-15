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

from fets.data.pytorch import new_labels_from_float_output
from fets.data.pytorch.gandlf_data import GANDLFData
from fets.models.pytorch.brainmage import BrainMaGeModel
from fets.models.pytorch.brainmage.losses import fets_phase2_validation

from GANDLF.utils import one_hot




################################################################
# Hard coded parameters (Make sure these apply for your model) #
################################################################

# Note the dependency on converting outputs back to class labels (we check for this to be consistent with plan used)
class_label_map = {0:0, 1:1, 2:2, 4:4}
class_list = list(np.sort(list(class_label_map.values())))
# data has shape 240, 240, 155, we need to pad the z axis in  order to reach divisibility by 16 in all dimensions
pad_z = 5

def is_mask_present(subject_dict):
    first = next(iter(subject_dict['label']))
    if first == 'NA':
        return False
    else:
        return True
    

def subject_to_feature_and_label(subject, pad_z):
    features = torch.cat([subject[key][torchio.DATA] for key in ['1', '2', '3', '4']], dim=1)
    
    if is_mask_present(subject):
        label = subject['label'][torchio.DATA]
    else:
        label = None
    
    if pad_z != 0:
        features_pad = torch.zeros(1, 4, 240, 240, pad_z)
        features = torch.cat([features, features_pad], dim=4)

        # we pad only the features, not the label
        # the output = model(features) will be cropped to restore its shape
        
    print("Constructed features from subject with shape", features.shape)
    if label is not None:
        print("Constructed label from subject with shape",label.shape)
        
    return features, label


def infer(model, _input):
    model.eval()
    with torch.no_grad():
        return model(_input.to(torch.device(model.device)))

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

    # TODO: We do not currently make use of the ability for brainmage to infer by first cropping external
    #       zero planes, or inference by patching and fusing.

    flplan = parse_fl_plan(plan_path)

    # make sure the class list we are using is compatible with the hard-coded class_label_map above
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

    subdir_to_DICE = {}
    dice_outpath = None

    for subject in data.get_val_loader():
        first_mode_path = subject['1']['path'][0] # using this because this is only one that's always defined
        subfolder = first_mode_path.split('/')[-2]
        
        #prep the path for the output files
        output_subdir = os.path.join(output_pardir, subfolder)
        if not os.path.exists(output_subdir):
            os.mkdir(output_subdir)
        inference_outpath = os.path.join(output_subdir, subfolder + model_output_tag + '_seg.nii.gz')
        if dice_outpath == None:
            dice_outpath = os.path.join(output_pardir, model_output_tag + '_subdirs_to_DICE.pkl')

        if not is_mask_present(subject):
            raise ValueError('We are expecting to run this on subjects that have labels.')

        label_path = subject['label']['path'][0]
        label_file = label_path.split('/')[-1]
        subdir_name = label_path.split('/')[-2]
        
        # copy the label file over to the output subdir
        copy_label_path = os.path.join(output_subdir, label_file)
        shutil.copyfile(label_path, copy_label_path)


        
        features, ground_truth = subject_to_feature_and_label(subject=subject, pad_z=pad_z)
                                    
        output = infer(model, features)

        # FIXME: Find a better solution
        # crop away the padding we put in
        output =  output[:, :, :, :, :155]

        print(one_hot(segmask_array=ground_truth, class_list=class_list).shape, output.shape)

        # get the DICE score
        dice_dict = fets_phase2_validation(output=output, 
                                  target=one_hot(segmask_array=ground_truth, class_list=class_list), 
                                  class_list=class_list, 
                                  to_scalar=True)

        subdir_to_DICE[subdir_name] = dice_dict

        output = np.squeeze(output.cpu().numpy())

        # GANDLFData loader produces transposed output from what sitk gets from file, so transposing here.
        output = np.transpose( output, [0, 3, 2, 1])

        # process float outputs (accros output channels), providing labels as defined in values of self.class_label_map
        output = new_labels_from_float_output(array=output,
                                              class_label_map=class_label_map, 
                                              binary_classification=False)
        
        # convert array to SimpleITK image 
        image = sitk.GetImageFromArray(output)

        image.CopyInformation(sitk.ReadImage(first_mode_path))

        print("\nWriting inference NIfTI image of shape {} to {}".format(output.shape, inference_outpath))
        sitk.WriteImage(image, inference_outpath)
        print("\nCorresponding DICE scores were: ")
        print("{}\n\n".format(dice_dict))

    print("Saving subdir_name_to_DICE at: ", dice_outpath)
    with open(dice_outpath, 'wb') as _file:
        pkl.dump(subdir_to_DICE, _file)

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

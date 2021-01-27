import numpy as np
import os
import argparse
import math
import shutil

import matplotlib.pyplot as plt

from fets.data.pytorch.gandlf_data import GANDLFData
from fets.models.pytorch.pt_3dresunet.pt_3dresunet import PyTorch3DResUNet as Model
from fets.data.pytorch import new_labels_from_float_output

import SimpleITK as sitk

from openfl import split_tensor_dict_for_holdouts
from openfl.tensor_transformation_pipelines import NoCompressionPipeline
from openfl.proto.protoutils import deconstruct_proto, load_proto


import torch
import torchio


################################################################
# Hard coded parameters (Make sure these apply for your model) #
################################################################

# Note the dependency on choice of n_classes and n_channels for model constructor below
class_label_map = {0:0, 1:1, 2:2, 4:4}

def is_mask_present(subject_dict):
    first = next(iter(subject_dict['label']))
    if first == 'NA':
        return False
    else:
        return True
    

def subject_to_feature_and_label(subject, pad_z=5):
    features = torch.cat([subject[key][torchio.DATA] for key in ['1', '2', '3', '4']], dim=1)
    print(features.shape)
    
    if is_mask_present(subject):
        label = subject['label'][torchio.DATA]
        print(label.shape)
    else:
        label = None
    
    if pad_z != 0:
        features_pad = torch.zeros(1, 4, 240, 240, pad_z)
        features = torch.cat([features, features_pad], dim=4)

        if label is not None:
            label_pad = torch.zeros(1, 1, 240, 240, pad_z)
            label = torch.cat([label, label_pad], dim=4)
        
    print("Constructed features from subject with shape", features.shape)
    if label is not None:
        print("Constructed label from subject with shape",label.shape)
        
    return features, label


def infer(model, _input):
    model.eval()
    with torch.no_grad():
        return model(_input.to(torch.device(model.device)))


#############################################################################
# Main will require a virtual environment with Algorithms, FeTS, and OpenFL #
#############################################################################
def main(data_csv_path, 
         gandlf_config_path, 
         model_weights_path, 
         output_pardir, 
         model_output_tag,
         device):

    # we will use the GANDLFData val loader to serve up the samples to perform inference on
    # will copy the data into the training loader (but not used)

    # These get passed to data constructor
    data_path = {'train': data_csv_path,
                 'val': data_csv_path,
                 'model_params_filepath': gandlf_config_path}
    divisibility_factor = 16

    # construct the data object
    data = GANDLFData(data_path=data_path, divisibility_factor=divisibility_factor)

    # construct the model object (requires cpu since we're passing [padded] whole brains)
    model = Model(data=data, 
                  base_filters = 30,
                  min_learning_rate = 0.000001,
                  max_learning_rate = 0.001,
                  learning_rate_cycles_per_epoch = 0.5,
                  n_classes=4,
                  n_channels=4,
                  loss_function = 'dc',
                  opt = 'sgd',
                  use_penalties = False, 
                  device=device)

    # Populate the model weights
    
    proto_path = model_weights_path
    proto = load_proto(proto_path)
    tensor_dict_from_proto = deconstruct_proto(proto, NoCompressionPipeline())
    # restore any tensors held out from the proto
    _, holdout_tensors = shared_tensors, holdout_tensors = split_tensor_dict_for_holdouts(None, model.get_tensor_dict())        
    tensor_dict = {**tensor_dict_from_proto, **holdout_tensors}
    model.set_tensor_dict(tensor_dict, with_opt_vars=False) 

    print("\nWill be running inference on {} samples.\n".format(model.get_validation_data_size()))

    if not os.path.exists(output_pardir):
        os.mkdir(output_pardir)

    for subject in data.get_val_loader():
        first_mode_path = subject['1']['path'][0] # using this because this is only one that's always defined
        subfolder = first_mode_path.split('/')[-2]
        
        #prep the path for the output file
        output_subdir = os.path.join(output_pardir, subfolder)
        if not os.path.exists(output_subdir):
            os.mkdir(output_subdir)
        outpath = os.path.join(output_subdir, subfolder + model_output_tag + '_seg.nii.gz')
        
        if is_mask_present(subject):
            label_path = subject['label']['path'][0]
            label_file = label_path.split('/')[-1]
            # copy the label file over to the output subdir
            copy_label_path = os.path.join(output_subdir, label_file)
            shutil.copyfile(label_path, copy_label_path)
        
        features, labels = subject_to_feature_and_label(subject)
                                    
        output = infer(model, features)
        
        output = np.squeeze(output.cpu().numpy())
        
        # crop away the padding we put in
        output =  output[:, :, :, :155]
        
        # the label on disk is transposed from what the gandlf loader produces
        print("\nWARNING: gandlf loader produces transposed output from what sitk gets from file, so transposing here.\n")
        output = np.transpose( output, [0, 3, 2, 1])

        # process float outputs (accros output channels), providing labels as defined in values of self.class_label_map
        output = new_labels_from_float_output(array=output,
                                              class_label_map=class_label_map, 
                                              binary_classification=False)

        # convert array to SimpleITK image 
        image = sitk.GetImageFromArray(output)

        image.CopyInformation(sitk.ReadImage(first_mode_path))

        print("\nWriting inference NIfTI image of shape {} to {}\n".format(output.shape, outpath))
        sitk.WriteImage(image, outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv_path', '-dcp', type=str, required=True)
    parser.add_argument('--gandlf_config_path', '-gcp', type=str, required=True)
    parser.add_argument('--model_weights_path', '-mwp', type=str, required=True)
    parser.add_argument('--output_pardir', '-op', type=str, default='./for_sarthak', required=True)
    parser.add_argument('--model_output_tag', '-mot', type=str, default='brandon_6', required=True)
    parser.add_argument('--device', '-dev', type=str, default='cpu', required=False)
    args = parser.parse_args()
    main(**vars(args))

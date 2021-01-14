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
import os

from torch.utils.data import DataLoader
import SimpleITK as sitk

from openfl import load_yaml
from fets.data.pytorch import TumorSegmentationDataset, check_for_file_or_gzip_file, find_file_or_with_extension, new_labels_from_float_output
from openfl.data.pytorch.ptfldata_inmemory import PyTorchFLDataInMemory

from fets.data import get_appropriate_file_paths_from_subject_dir
from fets.data.pytorch.file_lists_data import get_inference_dir_paths, remove_incomplete_data_paths, get_train_and_val_dir_paths

from fets.data import get_appropriate_file_paths_from_subject_dir


class PyTorchBrainMaGeData(PyTorchFLDataInMemory):

    def __init__(self, 
                 batch_size,
                 patch_size,
                 feature_modes,
                 data_path,
                 divisibility_factor,
                 class_label_map,
                 percent_train=0.8, 
                 label_tags = ["_seg_binary.nii", "_seg_binarized.nii", "_SegBinarized.nii", "_seg.nii"],
                 inference_patient = None,
                 **kwargs):

        super().__init__(batch_size=batch_size)

        self.psize = np.array(patch_size)
        self.feature_modes = feature_modes
        self.n_channels = len(feature_modes)
        self.feature_shape = self.n_channels
        # For loading inference data, where not only the patch size cannot ensure proper dimensions
        # but zero padding may also be needed.
        self.divisibility_factor = divisibility_factor

        self.label_tags = label_tags
        # dictionary conversion of loaded pixel labels (example: turn all classes 1,2, 4 into a 1)
        self.class_label_map = class_label_map

        # there is an assumption that the background class, 0, maps to 0 with no others doing so
        if class_label_map.get(0) is None:
            raise ValueError("class_label_map must contain the background zero class as a key")
        for key, label in class_label_map.items():
            if key == 0:
                if label != 0:
                    raise ValueError("class_label_map must send zero to zero")
            elif label == 0:
                raise ValueError("class_label_map is not allowed to send non-zero labels to zero")
        

        self.n_classes = len(np.unique(list(class_label_map.values())))

        # if we are performing binary classification per pixel, we will disable one_hot conversion of labels
        self.binary_classification =  self.n_classes == 2

        # there is an assumption that for binary classification, new classes are exactly 0 and 1
        if self.binary_classification:
            if set(class_label_map.values()) != set([0, 1]):
                raise ValueError("When performing binary classification, the new labels should be 0 and 1")
    

        
        self.train_dir_paths, self.val_dir_paths = get_train_and_val_dir_paths(data_path=data_path,
                                                                               feature_modes=self.feature_modes, 
                                                                               label_tags=self.label_tags, 
                                                                               percent_train=percent_train)
        
        self.inference_dir_paths = get_inference_dir_paths(data_path=data_path, feature_modes=self.feature_modes, inference_patient=inference_patient)

        self.inference_loader = self.create_loader(use_case="inference")
        self.train_loader = self.create_loader(use_case="training")
        self.val_loader = self.create_loader(use_case="validation")
        
        self.training_data_size = len(self.train_loader)
        self.validation_data_size = len(self.val_loader)

    def create_loader(self, use_case):
        if use_case == "training":
            dir_paths = self.train_dir_paths
            shuffle = True
        elif use_case == "validation":
            dir_paths = self.val_dir_paths
            shuffle = False
        elif use_case == "inference":
            dir_paths = self.inference_dir_paths
            shuffle = False
        else:
            raise ValueError("Specified use case for data loader is not known.")

        if len(dir_paths) == 0:
            return []
        else:
            dataset = TumorSegmentationDataset(dir_paths = dir_paths,
                                            feature_modes = self.feature_modes, 
                                            label_tags = self.label_tags, 
                                            use_case=use_case,
                                            psize=self.psize,
                                            class_label_map = self.class_label_map,
                                            binary_classification = self.binary_classification, 
                                            divisibility_factor=self.divisibility_factor)
            return DataLoader(dataset,batch_size= self.batch_size,shuffle=shuffle,num_workers=1)

    
    def write_outputs(self, outputs, metadata, output_file_tag):
        for idx, output in enumerate(outputs):
            dir_path = metadata["dir_path"][idx]
            base_fname = os.path.basename(dir_path)
            fpath = os.path.join(dir_path, base_fname + "_" + output_file_tag + ".nii.gz")
            
            # process float outputs (accros output channels), providing labels as defined in values of self.class_label_map
            output = new_labels_from_float_output(array=output,class_label_map=self.class_label_map, binary_classification=self.binary_classification)
            
            # recovering from the metadata what the oringal input shape was
            original_input_shape = []
            original_input_shape.append(metadata["original_x_dim"].numpy()[idx])
            original_input_shape.append(metadata["original_y_dim"].numpy()[idx])
            original_input_shape.append(metadata["original_z_dim"].numpy()[idx])
            slices = [slice(0,original_input_shape[n]) for n in range(3)]

            # now crop to original shape (dependency on how original zero padding was done)
            output = output[tuple(slices)]

            # convert array to SimpleITK image 
            image = sitk.GetImageFromArray(output)
              
            # get header info from an input image
            allFiles = get_appropriate_file_paths_from_subject_dir(dir_path)
            input_image_fpath = allFiles['T1']
            input_image_fpath = find_file_or_with_extension(input_image_fpath)
            input_image= sitk.ReadImage(input_image_fpath)
            image.CopyInformation(input_image)

            print("Writing inference NIfTI image of shape {} to {}".format(output.shape, fpath))
            sitk.WriteImage(image, fpath)
            




        



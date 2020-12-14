"""
This module helps run inference for the nnUNet model.
Please read the following paper to learn more:
Fabian Isensee, Paul F. Jäger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein 
"Automated Design of Deep Learning Methods for Biomedical Image Segmentation" arXiv preprint arXiv:1904.08128 (2020).

Code below is modified from a docker image found at: https://hub.docker.com/r/fabianisensee/isen2020, which is
derived from code within the github repository: https://github.com/MIC-DKFZ/nnUNet, whose modules contain the following headers:

    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
# TODO: check that header is appropriate.





import os
import numpy as np
import shutil

import SimpleITK as sitk

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join 
from nnunet.inference.ensemble_predictions import merge
from nnunet.inference.predict import predict_cases


def apply_brats_threshold(fname, output_fname, threshold, replace_with):
    img_itk = sitk.ReadImage(fname)
    img_npy = sitk.GetArrayFromImage(img_itk)
    num_enh = np.sum(img_npy == 3)
    if num_enh < threshold:
        print(fname, "had only %d enh voxels, those are now necrosis" % num_enh)
        img_npy[img_npy == 3] = replace_with
    img_itk_postprocessed = sitk.GetImageFromArray(img_npy)
    img_itk_postprocessed.CopyInformation(img_itk)
    sitk.WriteImage(img_itk_postprocessed, output_fname)


def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg


def load_model_for_fold(self, fold):
    self.trainer.output_folder = os.path.join(self.params_folder, fold)
    self.trainer.load_best_checkpoint(train=False)


def load_convert_save(filename):
    a = sitk.ReadImage(filename)
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, filename) 


# wrap Fabian's inference code in a class, implementing the run_inference_and_store_results method
class NNUnetInferenceOnlyModel():
    def __init__(self, 
                 *args, 
                 data,
                 native_model_weights_filepath,
                 replace_with = 2, 
                 model_list = ['nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5',
                               'nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5',
                               'nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5'],
                 folds_list = [tuple(np.arange(5)), tuple(np.arange(5)), tuple(np.arange(15))],
                 threshold = 200,
                 **kwargs):
        """
        Instantiates and configures the high-level model object. Relies on multiple (sub)model architectures (and model instances).
        Population of weights (for these multiple models) is handled automatically within the run_inference_and_store_results method.

        Args: ...

        Kwargs: 
        data (OpenFederatedLearning fldata object)  : Produces the inference data loader (validation and training loader can be empty)
        native_model_weights_filepath (string)      : Where to look for the serialized models
        replace_with                                : Necrosis end non-enhancing tumor in Fabian's label convention (apply postprocessing before converting to brats labels!) 
        model_list (list of string)                 : Sub-models used during inference
        folds_list (list of iterable)               : Validation folds which led to model instances to consider during inference for
                                                      each submodel (note depency on model_list)
        threshold                                   : 
        kwargs (dictionary)                         : Passed to trainer constructor
        
        """

        self.data = data
        self.params_folder = native_model_weights_filepath 
        self.intermediate_out_folder = self.data.data_path 
        self.replace_with =  replace_with                             
        self.model_list = model_list               
        self.folds_list =  folds_list             
        self.threshold = threshold 
    
    def run_inference_and_store_results(self,output_file_tag=''):
        output_file_base_name = output_file_tag + "_nnunet_seg.nii.gz"
        
        # passing only lists of length one to predict_cases
        for inner_list in self.data.inference_loader:
            list_of_lists = [inner_list]
            
            # output filenames (list of one) include information about patient folder name
            # infering patient folder name from all file paths for a sanity check
            # (should give the same answer)
            folder_names = [fpath.split('/')[-2] for fpath in inner_list]
            if set(folder_names) != set(folder_names[:1]):
                raise RuntimeError('Patient file paths: {} were found to come from different folders against expectation.'.format(inner_list)) 
            patient_folder_name = folder_names[0]
            output_filename = patient_folder_name + output_file_base_name
            
            final_out_folder = join(self.intermediate_out_folder, patient_folder_name)

            intermediate_output_folders = []
            
            for model_name, folds in zip(self.model_list, self.folds_list):
                output_model = join(self.intermediate_out_folder, model_name)
                intermediate_output_folders.append(output_model)
                intermediate_output_filepaths = [join(output_model, output_filename)]
                maybe_mkdir_p(output_model)
                params_folder_model = join(self.params_folder, model_name)
                
                predict_cases(model=params_folder_model, 
                            list_of_lists=list_of_lists, 
                            output_filenames=intermediate_output_filepaths, 
                            folds=folds, 
                            save_npz=True, 
                            num_threads_preprocessing=1, 
                            num_threads_nifti_save=1, 
                            segs_from_prev_stage=None, 
                            do_tta=True, 
                            mixed_precision=True,
                            overwrite_existing=True, 
                            all_in_gpu=False, 
                            step_size=0.5)

            merge(folders=intermediate_output_folders, 
                output_folder=final_out_folder, 
                threads=1, 
                override=True, 
                postprocessing_file=None, 
                store_npz=False)

            f = join(final_out_folder, output_filename)
            apply_brats_threshold(f, f, self.threshold, self.replace_with)
            load_convert_save(f)

        _ = [shutil.rmtree(i) for i in intermediate_output_folders]

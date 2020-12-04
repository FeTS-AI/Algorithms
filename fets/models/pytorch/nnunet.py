# FIXME: Proper header
# TODO: Fill in header





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
                 model_device='cpu', 
                 algorithm_identifier = "isen2020",
                 separate_out_folder = None,
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
        algorithm_identifier (string)               : Goes into the parameter file that is collected
        separate_out_folder (string)                : Where to output results of inference if different from where we get the features
        native_model_weights_filepath (string)      : Where to look for the serialized models
        model_device (string)                       : TODO
        replace_with                                : Necrosis end non-enhancing tumor in Fabian's label convention (apply postprocessing before converting to brats labels!) 
        model_list (list of string)                 : Sub-models used during inference
        folds_list (list of iterable)               : Validation folds which led to model instances to consider during inference for
                                                      each submodel (note depency on model_list)
        threshold                                   : 
        kwargs (dictionary)                         : Passed to trainer constructor
        
        """

        self.data = data
        self.params_folder = native_model_weights_filepath 
        self.model_device = model_device                 
        self.algorithm_identifier = algorithm_identifier           
        self.out_folder = separate_out_folder or self.data.data_path 
        self.replace_with =  replace_with                             
        self.model_list = model_list               
        self.folds_list =  folds_list             
        self.threshold = threshold

        maybe_mkdir_p(self.out_folder)
        
    def run_inference_and_store_results(self,output_file_tag=''):
        output_folders = []
        output_filename = output_file_tag + "tumor_{}_class.nii.gz".format(self.algorithm_identifier)
        for model_name, folds in zip(self.model_list, self.folds_list):
            output_model = join(self.out_folder, model_name)
            output_folders.append(output_model)
            maybe_mkdir_p(output_model)
            params_folder_model = join(self.params_folder, model_name)

            output_filenames = [join(output_model, output_file_tag + output_filename)]

            # This loop is only meant to have one iteration
            for list_of_lists_inner_elements_as_tuples in self.data.inference_loader:
                list_of_lists = [[tuple[0] for tuple in l] for l in list_of_lists_inner_elements_as_tuples]
                predict_cases(model=params_folder_model, 
                            list_of_lists=list_of_lists, 
                            output_filenames=output_filenames, 
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

        merge(folders=output_folders, 
              output_folder=self.out_folder, 
              threads=1, 
              override=True, 
              postprocessing_file=None, 
              store_npz=False)

        f = join(self.out_folder, output_filename)
        apply_brats_threshold(f, f, self.threshold, self.replace_with)
        load_convert_save(f)

        _ = [shutil.rmtree(i) for i in output_folders]

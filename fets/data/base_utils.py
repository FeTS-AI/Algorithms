import os 
import numpy as np

# FIXME: Look to all usages and fix the fact that we are inspecting files multiple times
#  for example we can optimize how we determine which patient directories to exlcude due to missing files 
def get_appropriate_file_paths_from_subject_dir(dir_path, 
                                                include_labels=False, 
                                                allowed_labelfile_endings=["_seg_binary.nii.gz", "_seg_binarized.nii.gz", "_SegBinarized.nii.gz", "_seg.nii.gz"], 
                                                excluded_labelfile_endings=[]):
    '''
    This function takes a subject directory as input and return a dictionary of the full paths to the modalities (BraTS-specific)
    '''
    filesInDir = os.listdir(dir_path)

    # FIXME: There is more than one place the list below is defined (example: gandlf_data)
    # Move to one location and ensure sync with feature_modes from the flplan
    brats_modes = ['T1', 'T2', 'FLAIR', 'T1CE']
    label_tag = 'Label'
    # acceptable file endings for each scanning modality
    mode_to_endings = {'T1': ['_t1.nii.gz'], 
                      'T2': ['_t2.nii.gz'], 
                      'FLAIR': ['_flair.nii.gz'], 
                      'T1CE': ['_t1ce.nii.gz', '_t1gd.nii.gz']} 
    return_dict = {type: None for type in brats_modes}
    if include_labels:
        return_dict[label_tag] = None

    for _file in filesInDir:
        fpath = os.path.abspath(os.path.join(dir_path,_file))
        # Is this file a valid feature mode (and not already found)?
        for mode in brats_modes:
            if np.any([_file.endswith(ending) for ending in mode_to_endings[mode]]):
                if return_dict[mode] is None:
                    return_dict[mode] = fpath
                else:
                    raise RuntimeError('Found two {} files in {} '.format(mode, dir_path))
        if include_labels:
            # Is this file a valid label (and not alreay found or in the excluded labelfile list)
            allowed_label = np.any([_file.endswith(ending) for ending in allowed_labelfile_endings])
            excluded_label = np.any([_file.endswith(ending) for ending in excluded_labelfile_endings])
            if allowed_label and not excluded_label:
                if return_dict[label_tag] is None:
                    return_dict[label_tag] = fpath
                else:
                    raise RuntimeError('Found two label files (allowing any of {} and excluding any of {}) in directory {} '.format(allowed_labelfile_endings, excluded_labelfile_endings, dir_path))

    for key, value in return_dict.items():
        if value is None:
            raise ValueError('No {} file found in {}.'.format(key, dir_path))
                 
    return return_dict

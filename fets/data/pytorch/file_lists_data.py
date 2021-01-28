# The following code is modified from code within this same repository, as well as from https://github.com/IntelLabs/OpenFederatedLearning. 
# TODO: check that header is sufficient.

import os
import numpy as np

from fets.data import get_appropriate_file_paths_from_subject_dir

def get_inference_dir_paths(data_path, feature_modes, inference_patient):
    inference_dir_paths = [os.path.join(data_path,dir_name) for dir_name in os.listdir(data_path)]
    if inference_patient is not None:
        new_paths = []
        for path in inference_dir_paths:
            if inference_patient in path:
                new_paths.append(path)
        inference_dir_paths = new_paths
    inference_dir_paths = remove_incomplete_data_paths(dir_paths = inference_dir_paths, feature_modes=feature_modes)
    return inference_dir_paths


def get_train_and_val_dir_paths(data_path, feature_modes, label_tags, percent_train):
    dir_names = os.listdir(data_path)
    dir_paths = [os.path.join(data_path, dir_name) for dir_name in dir_names]
    dir_paths = remove_incomplete_data_paths(dir_paths=dir_paths, 
                                            feature_modes=feature_modes, 
                                            label_tags=label_tags)
    dir_paths = np.random.permutation(dir_paths)
    index_cut = int(np.ceil(len(dir_paths) * percent_train))
    train_dir_paths, val_dir_paths = dir_paths[:index_cut], dir_paths[index_cut:]
    if set(train_dir_paths).union(set(val_dir_paths)) != set(dir_paths):
        raise ValueError("You have sharded data as to drop some or duplicate.")
    return train_dir_paths, val_dir_paths


def remove_incomplete_data_paths(dir_paths, feature_modes, label_tags=[]):
    filtered_dir_paths = []
    for path in dir_paths:
        dir_name = os.path.basename(path)
        # check to that all features are present
        all_modes_present = True
        allFiles = get_appropriate_file_paths_from_subject_dir(path)
        all_modes_present = all(allFiles.values())

        # for mode in feature_modes:
        #     fpath = os.path.join(path, dir_name + mode)
        #     if not os.path.exists(fpath):
        #         print("Path not present: ", fpath)
        #         all_modes_present = False
        #         break
        if all_modes_present:
            have_needed_labels = False
            for label_tag in label_tags:
                fpath = os.path.join(path, dir_name + label_tag)
                if os.path.exists(fpath):
                    have_needed_labels = True
                    break
            if label_tags == []:
                have_needed_labels = True
        
        if all_modes_present and have_needed_labels:
            filtered_dir_paths.append(path)
        else:
            print("Excluding data directory: {}, as not all required files present.".format(dir_name))
    return filtered_dir_paths


class FileListsData(object):
    """
    This is a very simple data object, only used to filter data subdirectories based on 
    insufficient files or by direction from parameters. Training and validation loaders
    are not defined, and the inference loader is a list over patients of filepath lists (over scan modalities). 
    """

    def __init__(self, 
                data_path,
                feature_modes=["_t1.nii.gz", "_t2.nii.gz", "_flair.nii.gz", "_t1ce.nii.gz"],
                inference_patient = None,
                **kwargs):


        self.data_path = data_path
        self.feature_modes = feature_modes

        self.inference_dir_paths = get_inference_dir_paths(data_path=self.data_path, feature_modes=feature_modes, inference_patient=inference_patient)

        self.inference_loader = self.get_inference_loader()

    def get_inference_loader(self):
        dir_paths = self.inference_dir_paths
        
        list_of_lists = []
        for dir_path in dir_paths:
            # filename matches last directory name

            allFiles = get_appropriate_file_paths_from_subject_dir(dir_path)
            # The order below is intentionally different from that of the brats modalities list to match
            # that used in nnunnet
            list_of_lists.append([allFiles['T1'], allFiles['T1CE'], allFiles['T2'], allFiles['FLAIR']])

        return list_of_lists



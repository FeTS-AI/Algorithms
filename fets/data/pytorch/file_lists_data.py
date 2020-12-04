# The following code is modified from https://github.com/CBICA/BrainMaGe which has the following lisence:

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

# FIXME: modify header to describe inlcusion of Fabian code
# TODO: modify header

import os

from torch.utils.data import Dataset, DataLoader

from batchgenerators.utilities.file_and_folder_operations import subfiles, join

from openfl.data.pytorch.ptfldata_inmemory import PyTorchFLDataInMemory

class FileListsDataset(Dataset):
    """
    This is a very simple dataset object.The __get_item__ function will only be called once,
    returning a list (over patients) of file lists (over modalities).
    """

    def __init__(self, dir_paths):
        self.dir_paths = dir_paths
        
    def __len__(self):
        return 1

    def __getitem__(self, index):

        if index !=0:
            raise ValueError('This dataset object is meant to have length one (serving up a list of lists of files).')

        list_of_lists = []
        for dir_path in self.dir_paths:
            file_basename = os.path.basename(dir_path) # filename matches last dirname
            
            t1_file = join(dir_path, file_basename + "_t1.nii.gz")
            t1c_file = join(dir_path, file_basename + "_t1ce.nii.gz")
            t2_file = join(dir_path, file_basename + "_t2.nii.gz")
            flair_file = join(dir_path, file_basename + "_flair.nii.gz")
            
            list_of_lists.append([t1_file, t1c_file, t2_file, flair_file])

        print("I'm in the datset object, and here is what the list of lists looks like: ", list_of_lists)
        
        return list_of_lists


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


def remove_incomplete_data_paths(dir_paths, feature_modes, label_tags=[]):
    filtered_dir_paths = []
    for path in dir_paths:
        dir_name = os.path.basename(path)
        # check to that all features are present
        all_modes_present = True
        for mode in feature_modes:
            fpath = os.path.join(path, dir_name + mode)
            if not os.path.exists(fpath):
                print("Path not present: ", fpath)
                all_modes_present = False
                break
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


class FileListsData(PyTorchFLDataInMemory):
    """
    This is a very simple data object, only used to filter data subdirectories based on 
    insufficient files or by direction from parameters. Training and validation loaders
    are empty, and the inference loader always has length one - returning a list over
    patients of a filepaths list over file modaities. A consequence of the quick use of
    the PyTorch data object is that the infal filepaths are a single element tuple of string
    rather than a string itself. I am simply modifying the inner filepaths from tuple to
    string when using this data object.
    """

    def __init__(self, 
                 data_path,
                 feature_modes=["_t1.nii.gz", "_t2.nii.gz", "_flair.nii.gz", "_t1ce.nii.gz"],
                 batch_size=1,
                 inference_patient = None,
                 **kwargs):

        super().__init__(batch_size=batch_size)

        self.data_path = data_path
        self.feature_modes = feature_modes

        self.inference_dir_paths = get_inference_dir_paths(data_path=self.data_path, feature_modes=feature_modes, inference_patient=inference_patient)

        self.inference_loader = self.create_loader(use_case="inference")
        
        self.train_dir_paths = None
        self.val_dir_paths = None
        self.training_data_size = 0
        self.validation_data_size = 0

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
            dataset = FileListsDataset(dir_paths = dir_paths)
        return DataLoader(dataset,batch_size= self.batch_size,shuffle=shuffle,num_workers=0)



# The following code is modified from ...
# FIXME: write header no Fabian pointers needed at this time, but maybe point to openFL (first functions)
# TODO: modify header

import os


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
            file_basename = os.path.basename(dir_path) 
            
            t1_file = os.path.join(dir_path, file_basename + "_t1.nii.gz")
            t1c_file = os.path.join(dir_path, file_basename + "_t1ce.nii.gz")
            t2_file = os.path.join(dir_path, file_basename + "_t2.nii.gz")
            flair_file = os.path.join(dir_path, file_basename + "_flair.nii.gz")
            
            list_of_lists.append([t1_file, t1c_file, t2_file, flair_file])

        return list_of_lists



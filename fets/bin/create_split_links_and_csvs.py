import argparse
import os
import numpy as np

from GANDLF.utils import writeTrainingCSV


def create_part(subfolder_names, actualfiles_dir, link_dirpath):
    # create the hyperlinks for one of the splits
    for folder_name in subfolder_names:
        symlink_path = os.path.join(link_dirpath, folder_name)
        actualfiles_dirpath = os.path.join(actualfiles_dir, folder_name)
        os.symlink(src=actualfiles_dirpath, dst=symlink_path, target_is_directory=True) 
    


def main(original_data_path, percent_train=0.8,split_dirname='TrainValSplits'):
    """
    Creates symlink directories for train and val data pointing to original_data_path, then
    creates GANDLF config csvs for each. Both the symlink directories and csvs will be placed
    in a folder in the same directory as sits original_data_path, and its name will be whatever
    is provided as split_dirname.
    NOTE: The channel ids and label id to look for in csv creation are hard-coded below.
          Though there is an order associated to the channels produced by the csv, it is the FeTS 
          gandlf_data object that specifies the order of modalities produced for the feature
          stacks of the final loaders!!!
    """

    if (percent_train <= 0.0) or (percent_train >= 1.0):
        raise ValueError("percent_train needs to be in (0,1) and not cause an empty train or val set")

    # get the list of data subdirectories
    subfolder_list = os.listdir(original_data_path)
    nb_subfolders = len(subfolder_list)
    np.random.shuffle(subfolder_list)
    print("\nWe have {} subfolders to split and create links and csvs for.\n".format(nb_subfolders))

    # split the list
    split_idx = int(percent_train * nb_subfolders) 
    train_subfolders = subfolder_list[:split_idx]
    val_subfolders = subfolder_list[split_idx:]
    nb_train = len(train_subfolders)
    nb_val = len(val_subfolders)
    if (nb_train == 0) or (nb_val == 0):
        raise ValueError("percent_train is not allowed to result in an emtpy train or val directory")
    print("Planning to create train folder of size {} and val folder of size {}, as percent_train is set to {}\n".format(nb_train, nb_val, percent_train))

    # create the train and val parent directories if they do not exist
    
    # normalize original_data_path
    original_data_path = os.path.normpath(original_data_path)
    # create output directory if it does not exist
    output_dir = os.path.join(os.path.split(original_data_path)[0], split_dirname)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_tag_int = int(percent_train * 100)
    val_tag_int = 100 - train_tag_int
    
    train_pardir = os.path.join(output_dir, 'Train' + str(train_tag_int) + 'SymbolicLinks')
    val_pardir = os.path.join(output_dir, 'Val' + str(val_tag_int) + 'SymbolicLinks')
    for path in [train_pardir, val_pardir]:
        if os.path.exists(path):
            raise RuntimeError("Script does not currently allow creating symlinks in an existing directory (can easily be changed).")
        else:
            os.mkdir(path)

    print("Creating symlinks for training directory: {}\n".format(train_pardir))
    create_part(subfolder_names=train_subfolders, 
                actualfiles_dir=original_data_path, 
                link_dirpath=train_pardir)
    print("Creating symlinks for validation directory: {}\n".format(val_pardir))
    create_part(subfolder_names=val_subfolders, 
                actualfiles_dir=original_data_path, 
                link_dirpath=val_pardir)
    
    # sanity checks
    nb_trainlinks = len(os.listdir(train_pardir))
    nb_vallinks = len(os.listdir(val_pardir))
    assert(nb_trainlinks + nb_vallinks == nb_subfolders)
    assert(nb_trainlinks - percent_train * nb_subfolders < 1)

    

    # Finally let's create csv files for the two new directories
    """
    NOTE: Though there is an order associated to the channels produced in the csv, it is the FeTS 
          gandlf_data object that specifies the order of modalities produced for the feature
          stacks of the final loaders!!!
    """
    channelsID = '_t1.nii.gz,_t2.nii.gz,_flair.nii.gz,_t1ce.nii.gz'
    labelID = '_seg.nii.gz'

    train_csvpath = os.path.join(output_dir, 'train_' + str(train_tag_int) + '.csv')
    print("Creating the training csv at path: {}\n".format(train_csvpath))
    writeTrainingCSV(channelsID=channelsID, 
                    labelID=labelID, 
                    inputDir=train_pardir, 
                    outputFile=train_csvpath)

    val_csvpath = os.path.join(output_dir, 'val_' + str(val_tag_int) + '.csv')
    print("Creating the validation csv at path: {}\n".format(val_csvpath))
    writeTrainingCSV(channelsID=channelsID, 
                    labelID=labelID, 
                    inputDir=val_pardir, 
                    outputFile=val_csvpath)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_data_path', '-odp', type=str, required=True)
    parser.add_argument('--percent_train', '-pt', type=float, default=0.8)
    parser.add_argument('--split_dirname', '-sd', type=str, default='TrainValSplits')
    args = parser.parse_args()
    main(**vars(args))


import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235

import sys
import pandas as pd

# TODO: replace function below with one that will soon be pushed to GANDLF

def get_dataframe_and_headers(file_data_full):
    """
    get the dataframe of subject filenames, and the headers of this dataframe.
    Parameters: 
    file_data_full (string): Data csv file containing paths to patient scans and labels (if appropriate)
    
    Copy of GANDLF.utils.parseTrainingCSV
  
    Returns: 
        int: Description of return value 
    """

    ## read training dataset into data frame
    data_full = pd.read_csv(file_data_full)
    # shuffle the data - this is a useful level of randomization for the training process
    data_full=data_full.sample(frac=1).reset_index(drop=True)

    # find actual header locations for input channel and label
    # the user might put the label first and the channels afterwards 
    # or might do it completely randomly
    headers = {}
    headers['channelHeaders'] = []
    headers['predictionHeaders'] = []
    headers['labelHeader'] = None
    headers['subjectIDHeader'] = None
    for col in data_full.columns: 
        # add appropriate headers to read here, as needed
        col_lower = col.lower()
        currentHeaderLoc = data_full.columns.get_loc(col)
        if ('channel' in col_lower) or ('modality' in col_lower) or ('image' in col_lower):
            headers['channelHeaders'].append(currentHeaderLoc)
        elif ('valuetopredict' in col_lower):
            headers['predictionHeaders'].append(currentHeaderLoc)
        elif ('subjectid' in col_lower) or ('patientname' in col_lower):
            headers['subjectIDHeader'] = currentHeaderLoc
        elif ('label' in col_lower) or ('mask' in col_lower) or ('segmentation' in col_lower) or ('ground_truth' in col_lower) or ('groundtruth' in col_lower):
            if (headers['labelHeader'] == None):
                headers['labelHeader'] = currentHeaderLoc
            else:
                print('WARNING: Multiple label headers found in training CSV, only the first one will be used', file = sys.stderr)
        
    return data_full, headers


"""
This is how above was used in gandlf_run
if mode != 0: # training mode
        TrainingManager(dataframe=data_full, headers = headers, outputDir=model_path, parameters=parameters, device=device)
    else:
        InferenceManager(dataframe=data_full, headers = headers, outputDir=model_path, parameters=parameters, device=device)
"""
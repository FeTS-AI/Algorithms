import os
import numpy as np
import nltk
import matplotlib.pyplot as plt


##########################################
# code to get raw arrays of round_numbers and valitation scores (but may have repeat entries)

def get_arrays_from_file(fpath, 
                        first_word='round', 
                        first_token_idx=1, 
                        sanity_check_token_idx=0, 
                        line_shift=2, 
                        second_token_idx=2):
    with open(fpath, 'r') as file:
        
        lines = file.readlines()
              
        # identify lines that start with first_word, and grab info from this line
        baseline_idxs = []
        round_nums = []
        for line_idx, line in enumerate(lines):
            tokens = nltk.word_tokenize(line)
            if (len(tokens) != 0) and (tokens[0] == first_word):
                baseline_idxs.append(line_idx)
                round_num = str(int(tokens[first_token_idx]) - 1)
                round_nums.append(round_num)
                
        
        # iterate over identified lines, shift lines and grab the second_token_idx token
        validation_scores = []
        for ls_idx, baseline_idx in enumerate(baseline_idxs):
            line_idx = baseline_idx + line_shift
            line = lines[line_idx]
            tokens = nltk.word_tokenize(line)
            # sanity check
            sanity_check_token = tokens[sanity_check_token_idx] 
            assert round_nums[ls_idx] in sanity_check_token
            validation_scores.append(tokens[second_token_idx])
        
        # cast strings into numbers
        round_nums = np.array(round_nums).astype(np.int)
        validation_scores = np.array(validation_scores).astype(np.float)
        
    return round_nums, validation_scores


def faster_get_arrays_from_file(fpath, 
                                first_word='round', 
                                first_token_idx=1, 
                                sanity_check_token_idx=0, 
                                line_shift=2, 
                                second_token_idx=2):
    with open(fpath, 'r') as file:
        
        lines = file.readlines()
        # line_tokens = [nltk.word_tokenize(line) for line in lines]
              
        # identify lines that start with first_word, and grab info from this line 
        # and also grab info from the line shifted away
        round_nums = []
        validation_scores = []
        for line_idx, line in enumerate(lines):
            if line.startswith(first_word):
                tokens = nltk.word_tokenize(line)
                shifted_line_idx = line_idx + line_shift
                round_num = str(int(tokens[first_token_idx]) - 1)
                round_nums.append(round_num)
                shifted_line_tokens = nltk.word_tokenize(lines[shifted_line_idx])
                if sanity_check_token_idx is not None:
                    # sanity check
                    sanity_check_token = shifted_line_tokens[sanity_check_token_idx] 
                    assert round_num in sanity_check_token
                validation_scores.append(shifted_line_tokens[second_token_idx])

        
        # cast strings into numbers
        round_nums = np.array(round_nums).astype(np.int)
        validation_scores = np.array(validation_scores).astype(np.float)
        
    return round_nums, validation_scores




##################################
# code to take the two lists: round_nums, validatation scores and remove repeat entries

def set_or_validate_same(dict, tuple):
    key, value = tuple
    if dict.get(key) is None:
        dict[key] = value
    elif dict[key] != value:
        raise ValueError("The log had the round number occur more than once with different validation scores.")
        
def unzip_to_numpy(dict):
    keys = []
    values = []
    for key in dict:
        keys.append(key)
        values.append(dict[key])
    return np.array(keys).astype(np.int), np.array(values).astype(np.float)
        

def remove_repeats(round_nums, validation_scores):
    round_val_dict = {}
    for round_val in zip(round_nums, validation_scores):
        set_or_validate_same(round_val_dict, round_val)
    return unzip_to_numpy(round_val_dict)


def faster_remove_repeats(round_nums, validation_scores):
    top_round = np.amax(round_nums)
    out_rounds = []
    out_vals = []
    for rnd in range(top_round+1):
        for idx, round_num in enumerate(round_nums):
            if round_num == rnd:
                out_rounds.append(round_num)
                out_vals.append(validation_scores[idx])
    return out_rounds, out_vals
            
        
        

########################################
# Code to bring together convergence info from all log files 

def parse_logs(running=False, specific_directory='Completed/agg_20201014-214253', faster=True, parent_dir = '/home/demo/demos/sc_braintumor_graphine_demo/fledge/sgx_demo/logs', info='val'):
    # info can be 'val' or 'loss'
    # an example of specific directory would be 'Completed/col_20201012-092333'

    if running:
        log_dir = os.path.join(parent_dir, 'Running')
        dirpath_list = []
        for name in os.listdir(log_dir):
            path = os.path.join(log_dir, name)
            if os.path.isdir(path) and ('agg' in name):
                dirpath_list.append(path)
    elif specific_directory == '':
        raise ValueError('If running is False, spefici_dirctory needs to be set.')
    else:
        dirpath_list = [os.path.join(parent_dir, specific_directory)]

    for p_idx, dirpath in enumerate(dirpath_list):
        fpath = os.path.join(dirpath, 'aggregator.log')
        print("Parsing: ", os.path.join(os.path.basename(dirpath), 'aggregator.log'))
        if faster:
            if info=='val':
                temp_round_nums, temp_scores = faster_get_arrays_from_file(fpath)
            elif info=='loss':
                temp_round_nums, temp_scores = faster_get_arrays_from_file(fpath, line_shift=1, sanity_check_token_idx=None)
            else:
                raise ValueError('info must be val or loss, not: ', info)
        else:
            if info == 'val':
                temp_round_nums, temp_scores = get_arrays_from_file(fpath)
            else:
                raise ValueError('info not val is not supported here')
        if p_idx == 0:
            round_nums = temp_round_nums
            scores = temp_scores
        else:
            round_nums = np.concatenate([round_nums, temp_round_nums])
            scores = np.concatenate([scores, temp_scores])

    if faster:
        round_nums, scores = faster_remove_repeats(round_nums, scores)
    else:
        round_nums, scores = remove_repeats(round_nums, scores)

    return round_nums, scores


#####################################################
#  Code to plot the convergence curve

def plot_convergence_curve(round_nums, 
                           validation_scores, 
                           text_font_size=80, 
                           label_size=40, 
                           line_width=10.0, 
                           info='val'):

    # figure = plt.figure()
    figure = plt.figure(figsize=(40, 25))
    _ = plt.tight_layout()
    ax = plt.subplot(111)
    _ = ax.plot(round_nums, validation_scores, linestyle='-', linewidth=line_width)
    if info == 'val':
        _ = ax.set_title('\nGlobal Model Validation \n', fontsize=text_font_size)
    elif info == 'loss':
        _ = ax.set_title('\nGlobal Model Loss \n', fontsize=text_font_size)
    else:
        raise ValueError('info must be val or loss')
    _ = ax.set_xlabel('Round', fontsize=text_font_size)
    _ = ax.set_ylabel('DICE', fontsize=text_font_size)
    if info == 'val':
        _ = ax.set(xlim=(0, np.amax(round_nums) + 10))
    elif info == 'loss':
        _ = ax.set(xlim=(0, np.amax(round_nums) + 10))
    _ = ax.tick_params(axis='x', labelsize=label_size)
    _ = ax.tick_params(axis='y', labelsize=label_size)
    _ = figure.show()

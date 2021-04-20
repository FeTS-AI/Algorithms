#!/usr/bin/env python3

import argparse
import sys
import os

from openfl.flplan import parse_fl_plan, create_data_object_with_explicit_data_path


def main(plan_path, data_path):
    """Creates the data split assosciated to a particular flplan using data found in data_dir

    Args:
        plan                            : The filename for the federation (FL) plan YAML file
        data_dir                        : parent directory holding the patient data subdirectories(to be split into train and val)
        
    """
    
    flplan = parse_fl_plan(os.path.join(plan_path))

    create_data_object_with_explicit_data_path(flplan, data_path=data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan_path', '-pp', type=str, required=True)
    parser.add_argument('--data_path', '-dp', type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))

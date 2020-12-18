#!bin/bash

# NOTES:
# Will need to run this in an activated virtual environment containing Algorithms, OpenFL, etc.
# Will need to define CUDA_VISIBLE_DEVICES once in the environment
# both stdout logs and usual OPenFL logs are written to file (see below or script printf output for where)
# There are two command line args, and one place to insert a hard coded path (to OpenFL)


# These are the command line arguments
START_IDX=$1
END_IDX=$2

# Did you forget a command line arg?
if [ -z "$START_IDX" ] || [ -z "$END_IDX" ] 
  then
   echo "No value of START_IDX and or END_IDX was provided"
   exit 1
fi

#################################################
#   Hard coded params you will need to modify.  #
#################################################
OFL_PATH="/home/edwardsb/repositories/be-OpenFederatedLearning"


# resulting paths built relative to above
RUNNING_LOG_DIR="${OFL_PATH}/bin/experiment_stdout_logs"
PATH_TO_RUN_SIMULATION="${OFL_PATH}/bin"

printf "${RUNNING_LOG_DIR}\n"

# create the stdout log dir if it does not exist
mkdir -p $RUNNING_LOG_DIR






for ((i=START_IDX;i<=END_IDX;i++));
do
    printf "\n\nStarting training with plan and config number ${i}" 
    printf "\n(NOTE: should be in your virtualenv with CUDA_VISIBLE_DEVICES defined)."
    printf "\nSTDOUT logs will go to: $RUNNING_LOG_DIR/stdout_idx_${i}.log"
    printf "\nOFL logs will go to: ${OFL_PATH}/bin/logs_${i} \n\n"
    cmd="python ${PATH_TO_RUN_SIMULATION}/run_simulation_from_flplan.py -p GANDLF_loader_pt_3dresunet_${i}.yaml -c pretraining_col.yaml -md cuda -ld ${OFL_PATH}/bin/logs_${i}"
    
    echo "Ran $cmd" 2>&1 | tee -a $RUNNING_LOG_DIR/stdout_idx_${i}.log 
    time $cmd 2>&1 | tee -a $RUNNING_LOG_DIR/stdout_idx_${i}.log
 
done


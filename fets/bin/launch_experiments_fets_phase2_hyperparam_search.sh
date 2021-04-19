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
PATH_TO_RUN_SIMULATION="${OFL_PATH}/bin"

for ((i=START_IDX;i<=END_IDX;i++));
do

    OFL_LOGDIR="${OFL_PATH}/bin/logs_phase2_${i}"
    PLAN="fets_phase2_${i}.yaml"
    printf "\n\nStarting training with plan ${PLAN}" 
    printf "\n(NOTE: should be in your virtualenv with CUDA_VISIBLE_DEVICES defined)."
    printf "\nOFL logs will go to: ${OFL_LOGDIR} \n\n"
    cmd="python ${PATH_TO_RUN_SIMULATION}/run_simulation_from_flplan.py -p ${PLAN} -c pretraining_col.yaml -md cuda -ld ${OFL_LOGDIR}"
     
    time $cmd 
 
done


#!/bin/sh
#
#$ -S /bin/bash

# TODO: Change this to your <ACTIVATE.SH> script!
source /data/mifs_scratch/fs439/exp/t2t/scripts/import_t2t_environment_cpu3.sh

if [ -z ${USR_DIR+x} ]; then T2TFlag=""; else T2TFlag="--t2t_usr_dir $USR_DIR"; fi

# This script requires the following variables (passed through by qsub via -v)
# SGE_TASK_ID: workers id (starting from 1)
# config_file: sgnmt config file
# output_dir: Output directory (will write to <out_dir>/SGE_TASK_ID

# Start decoding
mkdir -p $output_dir/$SGE_TASK_ID
python $SGNMT/decode.py --config_file $config_file --range $output_dir/remaining_ids $T2TFlag --single_cpu_thread true --output_path $output_dir/$SGE_TASK_ID/out.%s



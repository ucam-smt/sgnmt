#!/bin/sh
#
#$ -S /bin/bash

source /data/mifs_scratch/fs439/exp/t2t/scripts/import_t2t_environment_cpu3.sh

# This script requires the following variables (passed through by qsub via -v)
# SGE_TASK_ID: workers id (starting from 1)
# config_file: sgnmt config file
# output_dir: Output directory (will write to <out_dir>/SGE_TASK_ID

# Start decoding
mkdir -p $output_dir/$SGE_TASK_ID
python $SGNMT/decode.py --config_file $config_file --range $output_dir/remaining_ids --t2t_usr_dir $USR_DIR  --single_cpu_thread true --output_path $output_dir/$SGE_TASK_ID/out.%s



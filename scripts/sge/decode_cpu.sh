#!/bin/bash

if [ $# -ne 3 ] && [ $# -ne 4 ]; then
  echo "Usage: ./decode_cpu.sh <num-parts> <sgnmt-config-file> <output-dir> [<mem_required>]"
  echo "  This script distributes SGNMT decoding on the <num-parts> CPU nodes using the sun grid engine."
  echo "    <num-parts>: Number of jobs"
  echo "    <sgnmt-config-file>: Full SGNMT .ini file"
  echo "    <output-dir>: The outputs of each job will be written to <out-dir>/job-id/out.%s"
  echo "                  Combined files are created with the prefix <out-dir>/out.%s"
  echo "                  Logs are stored in <out-dir>/logs"
  echo "                  The output formats are specified via <sgnmt-config-file>"
  echo "    <mem_required>: Memory requirement (default: 1.5G)"
  exit 1;
fi

num_parts=$1
config_file=$2
output_dir=$3
worker_script=$(dirname $0)/decode_cpu_worker.sh
combination_script=$(dirname $0)/decode_combination.sh
mem_required="$4"
src_test="$(cat $config_file | egrep '^ *src_test *:' | cut -d':' -f2- | sed 's/ *//g')"
if [ -z "$(cat $config_file | egrep '^ *outputs *:.*nbest')" ]; then
  echo "You must use the nbest output format when using this script!"
  exit
fi
if [ -z "$mem_required" ]; then
  mem_required="1.5G"
fi

mkdir -p $output_dir/logs

# Start workers
echo "Decode "$(cat $src_test | wc -l)" sentences using $config_file with $num_parts workers ($mem_required), writing to $output_dir..."
cat $src_test | awk '{print NR" "NF}' | sort -g -k2 -r | cut -d' ' -f1 > $output_dir/remaining_ids


ID=$(qsub -N sgnmt-worker -l "mem_free=$mem_required,mem_grab=$mem_required,osrel=*,not_host=(air093|air094|air095|air096|air097|air098|air100|air101|air102|air106)" -o $output_dir/logs -e $output_dir/logs -t 1-$num_parts -v config_file=$config_file,output_dir=$output_dir $worker_script | awk '{print $3}' | sed 's:\..*::')
HT="-hold_jid $ID"

# Start combination job
qsub -N sgnmt-combination -l 'osrel=*' -o $output_dir/logs -e $output_dir/logs $HT -v output_dir=$output_dir,num_parts=$num_parts $combination_script


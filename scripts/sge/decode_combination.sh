#!/bin/sh
#
# the next line is a "magic" comment that tells gridengine to use bash
#$ -S /bin/bash
#
# and now for some real work

export LC_ALL=en_GB.utf8

# This script requires the variables output_dir set by qsub

rm -r $output_dir/out.text $output_dir/out.nbest $output_dir/out.fst $output_dir/out.sfst $output_dir/out.ngram $output_dir/out.timecsv

# Deal with nbest and text output format
cat $output_dir/*/out.nbest | sort -g -k1 -s > $output_dir/out.nbest
cat $output_dir/out.nbest | uniq -w 6 | cut -d'|' -f4 | sed 's/^ *//' | sed 's/ *$//' > $output_dir/out.text

rm $output_dir/*/out.text
rm $output_dir/*/out.nbest

for worker in $(ls $output_dir/ | egrep '^[0-9]+$')
do
  if [ -d $output_dir/$worker/out.fst ]
  then
    mkdir -p  $output_dir/out.fst
    cp $output_dir/$worker/out.fst/* $output_dir/out.fst
    rm -r $output_dir/$worker/out.fst
  fi
  if [ -d $output_dir/$worker/out.sfst ]
  then
    mkdir -p  $output_dir/out.sfst
    cp $output_dir/$worker/out.sfst/* $output_dir/out.sfst
    rm -r $output_dir/$worker/out.sfst
  fi
  if [ -d $output_dir/$worker/out.ngram ]
  then
    mkdir -p  $output_dir/out.ngram
    cp $output_dir/$worker/out.ngram/* $output_dir/out.ngram
    rm -r $output_dir/$worker/out.ngram
  fi
  if [ -d $output_dir/$worker/out.timecsv ]
  then
    mkdir -p  $output_dir/out.timecsv
    cp $output_dir/$worker/out.timecsv/* $output_dir/out.timecsv
    rm -r $output_dir/$worker/out.timecsv
  fi
  rmdir $output_dir/$worker
done

touch $output_dir/DONE

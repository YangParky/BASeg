#!/bin/sh

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp tool/testcityscapes.sh tool/testcityscapes.py ${config} ${exp_dir}

export PYTHONPATH=./
python -u ${exp_dir}/testcityscapes.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.log

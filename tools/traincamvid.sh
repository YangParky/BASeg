#!/bin/sh

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool/traincamvid.sh tool/traincamvid.py ${config} ${exp_dir}

export PYTHONPATH=./
python  -u ${exp_dir}/traincamvid.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train-$now.log

DATE=`date '+%Y-%m-%d-%H:%M:%S'`
title=main
# customize config path
config_path=configs/retrieved_data/pair_generation.json
# be consistent with your config_path
dataset=ecb # `fcc' or `gvc'
data_split=train # 'dev' or 'test'
random_seed=5
gpu_num=0
out_dir=retrieved_data/${title}

if [ ! -d "$out_dir" ];then
mkdir -p $out_dir
fi

echo "Retrieving nearest-K pairs"

nohup python -u src/all_models/generated_pairs.py --config_path ${config} --dataset ${dataset} --data_split ${data_split} --out_dir ${out_dir}\
    --random_seed ${random_seed} --gpu_num ${gpu_num} >${out_dir}/retrieving.log 2>${out_dir}/retrieving.progress &

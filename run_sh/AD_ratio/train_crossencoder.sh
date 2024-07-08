DATE=`date '+%Y-%m-%d-%H:%M:%S'`

gpu_num=0
random_seed=5
title=AD_ratio
dataset=fcc 
exp_name=fcc_events_ratio_0 # or `fcc_events_ratio_1', `fcc_events_ratio_3`, `fcc_events_ratio_5`
config_path=configs/${title}/${exp_name}.json
out_dir=outputs/${title}/${exp_name}/best_model

if [ ! -d "$out_dir" ];then
mkdir -p $out_dir
fi

# train crossencoder
echo "Train crossencoder"

nohup python -u src/all_models/crossencoder_trainer.py --config_path ${config_path}  --out_dir ${out_dir}\
    --mode train --random_seed ${random_seed} --gpu_num ${gpu_num} >${out_dir}/crossencoder.log 2>${out_dir}/crossencoder.progress &

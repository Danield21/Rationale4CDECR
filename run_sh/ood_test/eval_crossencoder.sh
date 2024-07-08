DATE=`date '+%Y-%m-%d-%H:%M:%S'`

gpu_num=0
random_seed=5
title=ood_test
dataset=ecb_cross_fcc #
exp_name=baseline # or `enhanced'
config_path=configs/${title}/${dataset}/${exp_name}.json
out_dir=outputs/${title}/${dataset}/${exp_name}/eval_results

if [ ! -d "$out_dir" ];then
mkdir -p $out_dir
fi

echo "Eval crossencoder"

nohup python -u src/all_models/crossencoder_trainer.py --config_path ${config_path} --out_dir ${out_dir}\
    --mode eval --random_seed ${random_seed} --gpu_num ${gpu_num} >${out_dir}/crossencoder.eval.log 2>${out_dir}/crossencoder.eval.progress &

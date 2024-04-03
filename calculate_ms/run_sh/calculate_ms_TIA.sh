#!/usr/bin/env bash
DATE=`date '+%Y-%m-%d-%H:%M:%S'`

data_name='TIA'
out_dir=outputs/${data_name}


if [ ! -d "$out_dir" ];then
    mkdir -p $out_dir
fi


echo "calculating ms on ${data_name}......"

nohup python -u calculate_ms.py -data ${data_name} -bz 200 -outdir ${out_dir} >${out_dir}/computing_moverscore.log 2>${out_dir}/computing_moverscore.progress& 

#!/bin/bash
export dataset="CiteSeer"
export im_class_num="3"
export im_ratio="0.05"
export cls_og="GNN"
export rec=True
export ep_pre=50
export class_weight=True
export gamma=1
export alpha=0.6
export lr_expert="0.01"
export embedder="lte4g"
export gpu=2

python main.py --dataset ${dataset} --im_class_num ${im_class_num} --im_ratio ${im_ratio} --cls_og ${cls_og} --rec ${rec} --ep_pre ${ep_pre} \
               --class_weight ${class_weight} --gamma ${gamma} --alpha ${alpha} --lr_expert ${lr_expert} --embedder ${embedder} --layer gcn --gpu ${gpu}



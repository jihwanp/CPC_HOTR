#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --project_name CPC_HOTR_VCOCO \
    --run_name ${EXP_DIR} \
    --HOIDet \
    --validate \
    --share_enc \
    --pretrained_dec \
    --use_consis \
    --share_dec_param \
    --epochs 90 \
    --lr_drop 60 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --ramp_up_epoch 30 \
    --path_id 0 \
    --num_hoi_queries 16 \
    --set_cost_idx 10 \
    --hoi_idx_loss_coef 1 \
    --hoi_act_loss_coef 10 \
    --backbone resnet50 \
    --hoi_consistency_loss_coef 1 \
    --hoi_idx_consistency_loss_coef 1 \
    --hoi_act_consistency_loss_coef 10 \
    --stop_grad_stage \
    --hoi_eos_coef 0.1 \
    --temperature 0.05 \
    --no_aux_loss \
    --hoi_aux_loss \
    --dataset_file vcoco \
    --data_path v-coco \
    --frozen_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output_dir checkpoints/vcoco/ \
    --augpath_name [\'p2\',\'p3\',\'p4\'] \
    ${PY_ARGS}
    
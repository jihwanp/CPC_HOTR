# CPC_HOTR

This repository contains the application of [Cross-Path Consistency Learning](https://arxiv.org/abs/2204.04836) at [HOTR](https://arxiv.org/abs/2104.13682), based on the official implementation of QPIC in [here](https://github.com/kakaobrain/HOTR).

<div align="center">
  <img src=".github/mainfig.png" width="900px" />
</div>


## 1. Environmental Setup
```bash
$ conda create -n HOTR_CPC python=3.7
$ conda install -c pytorch pytorch torchvision # PyTorch 1.7.1, torchvision 0.8.2, CUDA=11.0
$ conda install cython scipy
$ pip install pycocotools
$ pip install opencv-python
$ pip install wandb
```

## 2. HOI dataset setup
Our current version of HOTR supports the experiments for both [V-COCO](https://github.com/s-gupta/v-coco) and [HICO-DET](https://drive.google.com/file/d/1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk/view) dataset.
Download the dataset under the pulled directory.
For HICO-DET, we use the [annotation files](https://drive.google.com/file/d/1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk/view) provided by the PPDM authors.
Download the [list of actions](https://drive.google.com/open?id=1EeHNHuYyJI-qqDk_-5nay7Mb07tzZLsl) as `list_action.txt` and place them under the unballed hico-det directory.
Below we present how you should place the files.
```bash
# V-COCO setup
$ git clone https://github.com/s-gupta/v-coco.git
$ cd v-coco
$ ln -s [:COCO_DIR] coco/images # COCO_DIR contains images of train2014 & val2014
$ python script_pick_annotations.py [:COCO_DIR]/annotations

# HICO-DET setup
$ tar -zxvf hico_20160224_det.tar.gz # move the unballed folder under the pulled repository

# dataset setup
HOTR
 │─ v-coco
 │   │─ data
 │   │   │─ instances_vcoco_all_2014.json
 │   │   :
 │   └─ coco
 │       │─ images
 │       │   │─ train2014
 │       │   │   │─ COCO_train2014_000000000009.jpg
 │       │   │   :
 │       │   └─ val2014
 │       │       │─ COCO_val2014_000000000042.jpg
 :       :       :
 │─ hico_20160224_det
 │       │─ list_action.txt
 │       │─ annotations
 │       │   │─ trainval_hico.json
 │       │   │─ test_hico.json
 │       │   └─ corre_hico.npy
 :       :
```

If you wish to download the datasets on our own directory, simply change the 'data_path' argument to the directory you have downloaded the datasets.
```bash
--data_path [:your_own_directory]/[v-coco/hico_20160224_det]
```

## 3. Training
After the preparation, you can start the training with the following command.

For the HICO-DET training.
```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/hico_train.sh
```
For the V-COCO training.
```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/vcoco_train.sh
```

## 4. Evaluation
For evaluation of main inference path P1 (x->HOI), `--path_id` should be set to 0. 
Indexes of Augmented paths are range to 1~3. (1: x->HO->I, 2: x->HI->O, 3: x->OI->H)

HICODET
```
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --batch_size 2 \
    --HOIDet \
    --path_id 0 \
    --share_enc \
    --pretrained_dec \
    --share_dec_param \
    --num_hoi_queries [:query_num] \
    --object_threshold 0 \
    --temperature 0.2 \ # use the exact same temperature value that you used during training!
    --no_aux_loss \
    --eval \
    --dataset_file hico-det \
    --data_path hico_20160224_det \
    --resume checkpoints/hico_det/hico_[:query_num].pth
```

VCOCO
```
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env vcoco_main.py \
    --batch_size 2 \
    --HOIDet \
    --path_id 0 \
    --share_enc \
    --share_dec_param \
    --pretrained_dec \
    --num_hoi_queries [:query_num] \
    --temperature 0.05 \ # use the exact same temperature value that you used during training!
    --object_threshold 0 \
    --no_aux_loss \
    --eval \
    --dataset_file vcoco \
    --data_path v-coco \
    --resume checkpoints/vcoco/vcoco_[:query_num].pth
```

## Citation
```
@inproceedings{park2022consistency,
  title={Consistency Learning via Decoding Path Augmentation for Transformers in Human Object Interaction Detection},
  author={Park, Jihwan and Lee, SeungJun and Heo, Hwan and Choi, Hyeong Kyu and Kim, Hyunwoo J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
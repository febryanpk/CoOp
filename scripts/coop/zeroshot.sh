#!/bin/bash

# custom config
# DATA=/path/to/datasets
# DATA=/home/ryan/Desktop/HCC-Merlin/data_eurosat
# DATA=/home/ryan/Desktop/HCC-Merlin/data/food-101
DATA=${DATA:-/media/ryan/TOSHIBA2/nih_cxr}
TRAINER=ZeroshotCLIP2
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only
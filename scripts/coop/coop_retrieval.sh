#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=CoOp
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)

for SEED in 1 2 3
do 
  DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
  if [ -f "$DIR/retrieval_results.json" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
    python train.py \
      --root ${DATA} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/CoOp/${CFG}.yaml \
      --output-dir ${DIR} \
      --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
      --load-epoch 200 \
      --eval-only \
      --retrieval-eval \
      --retrieval-direction both \
      --retrieval-k 1 5 10
  fi
done
#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<data directory>"
PARENTNAME="videomae_pretrain_ccc"
PARENTRUN=1
NAME="videomae_ccc"
RUN=1
DESC="Train with all data and balanced cross entropy loss"

python train_videomae.py \
  --framedir "${DATADIR}/ntu_cataract/artifact/frame" \
  --framemetapath "${DATADIR}/ntu_cataract/artifact/metadata/annotated_frames.csv" \
  --sgmtmetapath "${DATADIR}/ntu_cataract/artifact/metadata/segment_annotation.csv" \
  --smplfreq 16 \
  --phase "ccc" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Grasp" \
  --action_group "Tear" \
  --clipsize 16 \
  --cliplength 1 \
  --clipstep 1 \
  --splitpath "${DATADIR}/ntu_cataract/artifact/metadata/video_split.csv" \
  --losstype "balanced_cross_entropy_loss" \
  --batchsize 8 \
  --num_workers 2 \
  --prtrnckptpath "${ROOTDIR}/model/${PARENTNAME}/run_${PARENTRUN}/best.ckpt" \
  --device "gpu" \
  --device_indices 0 \
  --num_epochs 40 \
  --num_accml_batches 4 \
  --logging_interval 20 \
  --mlflow_server_uri "file:${ROOTDIR}/logs/mlflow/mlruns" \
  --mlflow_expname "${NAME}" \
  --mlflow_runname "run_${RUN}" \
  --mlflow_rundesc "${DESC}" \
  --ckptdir "${ROOTDIR}/model/${NAME}/run_${RUN}" \
  --ckptname "best"

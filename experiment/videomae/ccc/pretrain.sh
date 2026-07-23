#!/bin/bash

set -e

ROOTDIR="<repository directory>"
NAME="videomae_pretrain_ccc"
RUN=1
DESC="Pre-train with all data"

python pretrain_videomae.py \
  --framedir "${ROOTDIR}/data/ntu_cataract/artifact/frame" \
  --framemetapath "${ROOTDIR}/data/ntu_cataract/artifact/metadata/annotated_frames.csv" \
  --sgmtmetapath "${ROOTDIR}/data/ntu_cataract/artifact/metadata/segment_annotation.csv" \
  --smplfreq 16 \
  --phase "ccc" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Cut" \
  --action_group "Grasp" \
  --action_group "Tear" \
  --clipsize 16 \
  --cliplength 1 \
  --clipstep 1 \
  --splitpath "${ROOTDIR}/data/ntu_cataract/artifact/metadata/video_split.csv" \
  --batchsize 8 \
  --num_workers 2 \
  --device "gpu" \
  --device_indices 0 \
  --num_epochs 100 \
  --num_accml_batches 4 \
  --logging_interval 20 \
  --mlflow_server_uri "file:${ROOTDIR}/logs/mlflow/mlruns" \
  --mlflow_expname "${NAME}" \
  --mlflow_runname "run_${RUN}" \
  --mlflow_rundesc "${DESC}" \
  --ckptdir "${ROOTDIR}/model/${NAME}/run_${RUN}" \
  --ckptname "best"

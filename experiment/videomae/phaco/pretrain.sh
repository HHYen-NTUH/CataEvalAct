#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<data direcotry>"
NAME="videomae_pretrain_phaco"
RUN=1
DESC="Pre-train with all data"

python pretrain_videomae.py \
  --framedir "${DATADIR}/ntu_cataract/artifact/frame" \
  --framemetapath "${DATADIR}/ntu_cataract/artifact/metadata/annotated_frames.csv" \
  --sgmtmetapath "${DATADIR}/ntu_cataract/artifact/metadata/segment_annotation.csv" \
  --smplfreq 4 \
  --phase "phaco" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Rotation" "Hook" \
  --action_group "Sculpt" \
  --action_group "Chop" "Separate" \
  --action_group "Engage" "Suction" "Pull" \
  --action_group "Hook and Suction" \
  --clipsize 16 \
  --cliplength 4 \
  --clipstep 1 \
  --clipfocalrange 1 2 \
  --splitpath "${DATADIR}/ntu_cataract/artifact/metadata/video_split.csv" \
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

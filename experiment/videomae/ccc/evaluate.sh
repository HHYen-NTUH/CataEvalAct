#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<data directory>"
NAME="videomae_ccc"
RUN=1
EVALNAME="videomae_by_phases"

python predict_videomae.py \
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
  --ckptpath "${ROOTDIR}/model/${NAME}/run_${RUN}/best.ckpt" \
  --batchsize 8 \
  --num_workers 2 \
  --device "gpu" \
  --device_indices 0 \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/prediction.csv"

python evaluate_action_prediction.py \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/prediction.csv" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Grasp" \
  --action_group "Tear" \
  --metricpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/metrics.csv" \
  --confmatpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/confusion_matrix.jpg"

# High-quality data

python predict_videomae.py \
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
  --score "four" "five" \
  --clipsize 16 \
  --cliplength 1 \
  --clipstep 1 \
  --splitpath "${DATADIR}/ntu_cataract/artifact/metadata/video_split.csv" \
  --ckptpath "${ROOTDIR}/model/${NAME}/run_${RUN}/best.ckpt" \
  --batchsize 8 \
  --num_workers 2 \
  --device "gpu" \
  --device_indices 0 \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/performance_4_5/prediction.csv"

python evaluate_action_prediction.py \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/performance_4_5/prediction.csv" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Grasp" \
  --action_group "Tear" \
  --metricpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/performance_4_5/metrics.csv" \
  --confmatpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/performance_4_5/confusion_matrix.jpg"

# Low-quality data

python predict_videomae.py \
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
  --score "one" "two" "three" "nan" \
  --exclude_idle \
  --clipsize 16 \
  --cliplength 1 \
  --clipstep 1 \
  --splitpath "${DATADIR}/ntu_cataract/artifact/metadata/video_split.csv" \
  --ckptpath "${ROOTDIR}/model/${NAME}/run_${RUN}/best.ckpt" \
  --batchsize 8 \
  --num_workers 2 \
  --device "gpu" \
  --device_indices 0 \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/performance_1_2_3_nan/prediction.csv"

python evaluate_action_prediction.py \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/performance_1_2_3_nan/prediction.csv" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Grasp" \
  --action_group "Tear" \
  --metricpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/performance_1_2_3_nan/metrics.csv" \
  --confmatpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/performance_1_2_3_nan/confusion_matrix.jpg"

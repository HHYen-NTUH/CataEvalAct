#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<data directory>"
NAME="videomae_phaco"
RUN=1
EVALNAME="videomae_by_phases"

python predict_videomae.py \
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
  --ckptpath "${ROOTDIR}/model/${NAME}/run_${RUN}/best.ckpt" \
  --batchsize 8 \
  --num_workers 2 \
  --device "gpu" \
  --device_indices 0 \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/prediction.csv"

python evaluate_action_prediction.py \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/prediction.csv" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Rotation" "Hook" \
  --action_group "Sculpt" \
  --action_group "Chop" "Separate" \
  --action_group "Engage" "Suction" "Pull" \
  --action_group "Hook and Suction" \
  --metricpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/metrics.csv" \
  --confmatpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/confusion_matrix.jpg"

# High-quality data

python predict_videomae.py \
  --framedir "${DATDIR}/ntu_cataract/artifact/frame" \
  --framemetapath "${DATDIR}/ntu_cataract/artifact/metadata/annotated_frames.csv" \
  --sgmtmetapath "${DATDIR}/ntu_cataract/artifact/metadata/segment_annotation.csv" \
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
  --score "four" "five" \
  --clipsize 16 \
  --cliplength 4 \
  --clipstep 1 \
  --clipfocalrange 1 2 \
  --splitpath "${DATDIR}/ntu_cataract/artifact/metadata/video_split.csv" \
  --ckptpath "${ROOTDIR}/model/${NAME}/run_${RUN}/best.ckpt" \
  --batchsize 8 \
  --num_workers 2 \
  --device "gpu" \
  --device_indices 0 \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/performance_4_5/prediction.csv"

python evaluate_action_prediction.py \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/performance_4_5/prediction.csv" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Rotation" "Hook" \
  --action_group "Sculpt" \
  --action_group "Chop" "Separate" \
  --action_group "Engage" "Suction" "Pull" \
  --action_group "Hook and Suction" \
  --metricpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/performance_4_5/metrics.csv" \
  --confmatpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/performance_4_5/confusion_matrix.jpg"

# Low-quality data

python predict_videomae.py \
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
  --score "one" "two" "three" "nan" \
  --exclude_idle \
  --clipsize 16 \
  --cliplength 4 \
  --clipstep 1 \
  --clipfocalrange 1 2 \
  --splitpath "${DATADIR}/ntu_cataract/artifact/metadata/video_split.csv" \
  --ckptpath "${ROOTDIR}/model/${NAME}/run_${RUN}/best.ckpt" \
  --batchsize 8 \
  --num_workers 2 \
  --device "gpu" \
  --device_indices 0 \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/performance_1_2_3_nan/prediction.csv"

python evaluate_action_prediction.py \
  --predpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/performance_1_2_3_nan/prediction.csv" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Rotation" "Hook" \
  --action_group "Sculpt" \
  --action_group "Chop" "Separate" \
  --action_group "Engage" "Suction" "Pull" \
  --action_group "Hook and Suction" \
  --metricpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/performance_1_2_3_nan/metrics.csv" \
  --confmatpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/performance_1_2_3_nan/confusion_matrix.jpg"

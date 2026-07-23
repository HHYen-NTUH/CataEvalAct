#!/bin/bash

set -e

ROOTDIR="<repository directory>"
EVALNAME="videomae_by_phases"
RUN=1

python evaluate_action_prediction.py \
  --predpath \
  "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/prediction.csv" \
  "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/prediction.csv" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Grasp" \
  --action_group "Tear" \
  --action_group "Rotation" "Hook" \
  --action_group "Sculpt" \
  --action_group "Chop" "Separate" \
  --action_group "Engage" "Suction" "Pull" \
  --action_group "Hook and Suction" \
  --metricpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/metrics.csv" \
  --confmatpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/confusion_matrix.jpg"

# High-quality data

python evaluate_action_prediction.py \
  --predpath \
  "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/performance_4_5/prediction.csv" \
  "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/performance_4_5/prediction.csv" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Grasp" \
  --action_group "Tear" \
  --action_group "Rotation" "Hook" \
  --action_group "Sculpt" \
  --action_group "Chop" "Separate" \
  --action_group "Engage" "Suction" "Pull" \
  --action_group "Hook and Suction" \
  --metricpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/performance_4_5/metrics.csv" \
  --confmatpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/performance_4_5/confusion_matrix.jpg"

# Low-quality data

python evaluate_action_prediction.py \
  --predpath \
  "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/ccc/performance_1_2_3_nan/prediction.csv" \
  "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/phaco/performance_1_2_3_nan/prediction.csv" \
  --action_group "Idle" \
  --action_group "In-Out wound" \
  --action_group "Recovery / Productive Adjustment" \
  --action_group "Grasp" \
  --action_group "Tear" \
  --action_group "Rotation" "Hook" \
  --action_group "Sculpt" \
  --action_group "Chop" "Separate" \
  --action_group "Engage" "Suction" "Pull" \
  --action_group "Hook and Suction" \
  --metricpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/performance_1_2_3_nan/metrics.csv" \
  --confmatpath "${ROOTDIR}/analysis/ntu_cataract/evaluation/${EVALNAME}/run_${RUN}/performance_1_2_3_nan/confusion_matrix.jpg"

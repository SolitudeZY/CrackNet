#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/fs-ai/miniconda3/envs/yolov11_py3.10/bin/python}"
PROJECT_DIR="/home/fs-ai/unet-crack/unet_seg_u2net"
DATASET_ROOT="${DATASET_ROOT:-/home/fs-ai/unet-crack/dataset/CRACK500}"
WEIGHTS_PATH="${WEIGHTS_PATH:?Please set WEIGHTS_PATH to your finetuned checkpoint}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" test.py \
  --weights "${WEIGHTS_PATH}" \
  --dataset-root "${DATASET_ROOT}" \
  --auto-threshold \
  --threshold-metric iou

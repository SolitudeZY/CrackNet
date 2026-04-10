#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/fs-ai/miniconda3/envs/yolov11_py3.10/bin/python}"
PROJECT_DIR="/home/fs-ai/unet-crack/unet_seg_u2net"
DATASET_ROOT="${DATASET_ROOT:-/home/fs-ai/unet-crack/dataset/CRACK500}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/home/fs-ai/unet-crack/U-2-Net/saved_models/u2net/u2net.pth}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" train.py \
  --model u2net \
  --dataset-root "${DATASET_ROOT}" \
  --pretrained \
  --pretrained-path "${PRETRAINED_PATH}" \
  --img-size 512 \
  --batch-size 4 \
  --grad-accumulation 2 \
  --epochs 180 \
  --lr 5e-5 \
  --loss boundary

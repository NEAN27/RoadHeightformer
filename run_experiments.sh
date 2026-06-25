#!/bin/bash

# RoadHeightFormer training launcher.
#
# Usage:
#   ./run_experiments.sh <config_file> [extra train.py args]
#
# Pick a GPU by exporting CUDA_VISIBLE_DEVICES (defaults to 0):
#   CUDA_VISIBLE_DEVICES=0 ./run_experiments.sh configs/config_freeze_baseline.yaml
#   CUDA_VISIBLE_DEVICES=1 ./run_experiments.sh configs/config_efficientnet.yaml

set -euo pipefail

CONFIG_FILE="${1:-configs/config_freeze_baseline.yaml}"
shift || true  # safe shift (no-op if no args)

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "====================="
echo "RoadHeightFormer run"
echo "====================="
echo "Config:                $CONFIG_FILE"
echo "CUDA_VISIBLE_DEVICES:  $CUDA_VISIBLE_DEVICES"
echo "Extra args:            $*"
echo "---------------------"

python train.py --config "$CONFIG_FILE" "$@"

# Examples:
#   CUDA_VISIBLE_DEVICES=0 ./run_experiments.sh configs/config_freeze_baseline.yaml
#   CUDA_VISIBLE_DEVICES=1 ./run_experiments.sh configs/config_efficientnet.yaml
#   CUDA_VISIBLE_DEVICES=0 ./run_experiments.sh configs/config_freeze_baseline.yaml --name_run "freeze_rerun"

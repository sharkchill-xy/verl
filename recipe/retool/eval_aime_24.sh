#!/bin/bash

# make sure your current working directory is the root of the project

set -x
ulimit -n 65535

model_path=/data2/lixy/VerlCoder/models/retool_sft_16epochs/global_step_30
data_path=/data2/lixy/VerlCoder/data/aime24_eval_multiturn/test.parquet
output_path=/data2/lixy/VerlCoder/results/aime24_generated_responses.parquet


PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/retool/config"

# 创建日志目录
LOG_DIR="/data2/lixy/VerlCoder/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/aime24_eval_$(date +%Y%m%d_%H%M%S).log"

echo "日志文件: $LOG_FILE"

python3 -m verl.trainer.main_generation \
    --config-path="$CONFIG_PATH" \
    --config-name=aime24_generation \
    model.path="$model_path" \
    data.path="$data_path" \
    data.output_path="$output_path" \
    data.n_samples=1 \
    data.batch_size=16 \
    data.return_raw_chat=true \
    rollout.n=1 \
    2>&1 | tee "$LOG_FILE"

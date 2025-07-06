#!/bin/bash

# 确保在正确的目录下运行
cd /home/lixy/workspace/VerlCoder/verl

# 设置环境变量
export PYTHONPATH="/home/lixy/workspace/VerlCoder/verl:$PYTHONPATH"

# 运行参数配置
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
BASE_URL="http://localhost:8000/v1"  # 假设使用 vLLM 或 SGLang 服务
API_KEY="test"
OUTPUT_PATH="/data2/lixy/VerlCoder/results/standalone_eval_results.json"

# Sandbox Fusion URL
SANDBOX_URL="http://210.28.135.36:8080"

# 评估参数
MAX_TURNS=4
N_SAMPLES=1
TEMPERATURE=0.7
TOP_P=0.9

echo "Starting standalone AIME24 evaluation..."
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_PATH"

python recipe/retool/eval_aime24_standalone.py \
    --model_name_or_path "$MODEL_NAME" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --output_path "$OUTPUT_PATH" \
    --sandbox_url "$SANDBOX_URL" \
    --max_turns $MAX_TURNS \
    --n_samples $N_SAMPLES \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_length 16384 \
    --sandbox_timeout 30

echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_PATH"
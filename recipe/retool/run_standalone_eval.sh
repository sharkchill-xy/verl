#!/bin/bash


# vllm serve /data2/lixy/VerlCoder/checkpoints/retool-multiturn-sft/retool-multiturn-sft-qwen2.5-7b-sp4-lr5e-6/global_step_42 \
# --port 8000 \
# --tensor-parallel-size 4 \ 
# --max-model-len 16384 \
# --enable-auto-tool-choice \
# --tool-call-parser hermes

# 确保在正确的目录下运行
cd /home/lixy/workspace/VerlCoder/verl

PYTHON_PATH=".venv/bin/python"

# 运行参数配置
MODEL_NAME="/data2/lixy/VerlCoder/checkpoints/retool-multiturn-sft/retool-multiturn-sft-qwen2.5-7b-sp4-lr5e-6/global_step_42"
BASE_URL="http://210.28.135.36:8000/v1"  
API_KEY="EMPTY"
OUTPUT_PATH="/data2/lixy/VerlCoder/results/standalone_eval_results.json"

# Sandbox Fusion URL
SANDBOX_URL="http://210.28.135.36:8080"

# 评估参数
MAX_TURNS=8
N_SAMPLES=32
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50

# 并发参数
MAX_CONCURRENT_REQUESTS=32
MAX_CONCURRENT_SANDBOX=32
REQUEST_TIMEOUT=300

echo "Starting standalone AIME24 evaluation..."
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_PATH"
echo "Samples per problem: $N_SAMPLES"
echo "Max concurrent requests: $MAX_CONCURRENT_REQUESTS"
echo "Max concurrent sandbox: $MAX_CONCURRENT_SANDBOX"

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
    --top_k $TOP_K \
    --max_length 16384 \
    --sandbox_timeout 30 \
    --max_concurrent_requests $MAX_CONCURRENT_REQUESTS \
    --max_concurrent_sandbox $MAX_CONCURRENT_SANDBOX \
    --request_timeout $REQUEST_TIMEOUT

echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_PATH"
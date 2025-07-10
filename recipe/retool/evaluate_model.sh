#!/bin/bash
# ReTool Model Evaluation Script
#
# This script evaluates a ReTool model on the AIME24 dataset using the standalone evaluation framework.
#
# Usage:
#   ./evaluate_model.sh [MODEL_PATH] [OUTPUT_PATH] [BASE_URL] [SANDBOX_URL]
#
# Arguments:
#   MODEL_PATH: Path to the model checkpoint to evaluate
#   OUTPUT_PATH: Path to save evaluation results JSON file
#   BASE_URL: API endpoint for model inference (default: http://localhost:8000/v1)
#   SANDBOX_URL: Sandbox Fusion API endpoint (default: http://localhost:8080)
#
# Example:
#   ./evaluate_model.sh /path/to/model /path/to/results.json http://server:8000/v1 http://server:8080

set -e

# Default values (can be overridden by command line arguments)
MODEL_NAME="${1:-/path/to/your/model}"
OUTPUT_PATH="${2:-./results/evaluation_results.json}"
BASE_URL="${3:-http://localhost:8000/v1}"
SANDBOX_URL="${4:-http://localhost:8080}"
API_KEY="${5:-EMPTY}"

# Validate inputs
if [[ "$MODEL_NAME" == "/path/to/your/model" ]]; then
    echo "Error: Please provide model path as first argument"
    echo "Usage: $0 <MODEL_PATH> <OUTPUT_PATH> [BASE_URL] [SANDBOX_URL] [API_KEY]"
    exit 1
fi

if [[ "$OUTPUT_PATH" == "./results/evaluation_results.json" ]]; then
    echo "Warning: Using default output path: $OUTPUT_PATH"
    mkdir -p "$(dirname "$OUTPUT_PATH")"
fi

echo "Starting ReTool Model Evaluation..."
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_PATH"
echo "Base URL: $BASE_URL"
echo "Sandbox URL: $SANDBOX_URL"

# Configuration
PROJECT_DIR="$(pwd)"

# 评估参数
MAX_TURNS=8
N_SAMPLES=32
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50

# 并发参数
MAX_CONCURRENT_REQUESTS=64
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
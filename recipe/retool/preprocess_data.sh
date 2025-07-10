#!/bin/bash
# ReTool Data Preprocessing Script
#
# This script preprocesses and filters the ReTool training dataset using the same filtering
# parameters that will be used during training to avoid redundant filtering.
#
# Usage:
#   ./preprocess_data.sh [INPUT_FILE] [OUTPUT_FILE] [TOKENIZER_PATH]
#
# Arguments:
#   INPUT_FILE: Path to the input parquet file to filter
#   OUTPUT_FILE: Path to save the filtered parquet file
#   TOKENIZER_PATH: Path to the tokenizer for length calculation
#
# Example:
#   ./preprocess_data.sh /path/to/train.parquet /path/to/train_filtered.parquet /path/to/tokenizer

set -e

# Default values (can be overridden by command line arguments)
INPUT_FILE="${1:-/path/to/your/input.parquet}"
OUTPUT_FILE="${2:-/path/to/your/output.parquet}"
TOKENIZER_PATH="${3:-/path/to/your/tokenizer}"

# Validate inputs
if [[ "$INPUT_FILE" == "/path/to/your/input.parquet" ]]; then
    echo "Error: Please provide input file path as first argument"
    echo "Usage: $0 <INPUT_FILE> <OUTPUT_FILE> <TOKENIZER_PATH>"
    exit 1
fi

if [[ "$OUTPUT_FILE" == "/path/to/your/output.parquet" ]]; then
    echo "Error: Please provide output file path as second argument"
    echo "Usage: $0 <INPUT_FILE> <OUTPUT_FILE> <TOKENIZER_PATH>"
    exit 1
fi

if [[ "$TOKENIZER_PATH" == "/path/to/your/tokenizer" ]]; then
    echo "Error: Please provide tokenizer path as third argument"
    echo "Usage: $0 <INPUT_FILE> <OUTPUT_FILE> <TOKENIZER_PATH>"
    exit 1
fi

# Ensure input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

echo "Starting ReTool Data Preprocessing..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Tokenizer: $TOKENIZER_PATH"

# Configuration
PROJECT_DIR="$(pwd)"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Activate virtual environment if it exists
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

# Default filtering parameters (can be customized as needed)
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
PROMPT_KEY="prompt"
NUM_WORKERS=192

# Run preprocessing script
python3 recipe/retool/prefilter_dataset.py \
    --input_file="$INPUT_FILE" \
    --output_file="$OUTPUT_FILE" \
    --tokenizer_path="$TOKENIZER_PATH" \
    --max_prompt_length=$MAX_PROMPT_LENGTH \
    --max_response_length=$MAX_RESPONSE_LENGTH \
    --prompt_key="$PROMPT_KEY" \
    --num_workers=$NUM_WORKERS

echo "✅ Data preprocessing completed!"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo ""
echo "You can now use the filtered dataset in training and disable filter_overlong_prompts"
echo "to skip filtering during training for improved performance."
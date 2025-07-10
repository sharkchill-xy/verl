#!/bin/bash
# 预过滤ReTool训练数据集
# 使用与训练时相同的过滤参数


PROJECT_DIR="$(pwd)"
INPUT_FILE="/data2/lixy/VerlCoder/data/retool_dapo_split/train.parquet"
OUTPUT_FILE="/data2/lixy/VerlCoder/data/retool_dapo_split/train_filtered.parquet"
TOKENIZER_PATH="/data2/lixy/VerlCoder/checkpoints/retool-multiturn-sft/retool-multiturn-sft-qwen2.5-7b-sp4-lr5e-6/global_step_42"

# 激活虚拟环境
source .venv/bin/activate

# 运行预过滤脚本
python3 recipe/retool/prefilter_dataset.py \
    --input_file="$INPUT_FILE" \
    --output_file="$OUTPUT_FILE" \
    --tokenizer_path="$TOKENIZER_PATH" \
    --max_prompt_length=2048 \
    --max_response_length=4096 \
    --prompt_key="prompt" \
    --num_workers=192

echo "✅ 预过滤完成！"
echo "原始文件: $INPUT_FILE"
echo "过滤后文件: $OUTPUT_FILE"
echo ""
echo "现在你可以在训练配置中使用 train_filtered.parquet 并关闭 filter_overlong_prompts 来跳过训练时的过滤步骤。"
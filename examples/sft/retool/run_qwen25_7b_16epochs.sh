#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen25_7b_16epochs.sh <nproc_per_node> <save_path> [other_configs...]"
    echo "Example: bash run_qwen25_7b_16epochs.sh 8 /data2/lixy/VerlCoder/models/retool_sft_16epochs"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# 确保数据目录存在
data_dir=/data2/lixy/VerlCoder/data/retool
if [ ! -d "$data_dir" ]; then
    echo "数据目录不存在: $data_dir"
    echo "请先运行数据预处理脚本: python verl/examples/data_preprocess/retool.py --local_dir $data_dir"
    exit 1
fi

# 检查数据文件是否存在
if [ ! -f "$data_dir/train.parquet" ] || [ ! -f "$data_dir/val.parquet" ]; then
    echo "数据文件不存在，请先运行数据预处理脚本"
    exit 1
fi

echo "开始 ReTool SFT 训练 - 16 epochs with validation"
echo "模型: Qwen2.5-7B-Instruct, ${nproc_per_node} GPUs"
echo "数据路径: $data_dir"
echo "保存路径: $save_path"
echo "训练设置: 16 epochs, 每epoch验证, 每epoch保存checkpoint"

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/val.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['prompt_text'] \
    +data.response_dict_keys=['response'] \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=4096 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=retool-sft \
    trainer.experiment_name=retool-sft-qwen2.5-7b-instruct-16epochs \
    trainer.total_epochs=16 \
    trainer.test_freq=6 \
    trainer.save_freq=6 \
    trainer.logger=['console','swanlab'] \
    trainer.default_hdfs_dir=null \
    $@

echo "16 epochs SFT训练完成！模型保存在: $save_path"
echo "每个epoch的checkpoint都已保存，可用于分析训练过程"
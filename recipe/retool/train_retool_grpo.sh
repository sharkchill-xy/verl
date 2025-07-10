#!/bin/bash
# ReTool GRPO Training Script
# 
# This script trains a ReTool model using GRPO (Generalized Reward Preference Optimization)
# for mathematical reasoning with code interpreter capabilities.
#
# Usage:
#   ./train_retool_grpo.sh [SFT_MODEL_PATH] [DATA_DIR] [OUTPUT_DIR]
#
# Arguments:
#   SFT_MODEL_PATH: Path to the SFT checkpoint to start from
#   DATA_DIR: Directory containing train.parquet and val.parquet  
#   OUTPUT_DIR: Directory to save checkpoints and logs
#
# Example:
#   ./train_retool_grpo.sh /path/to/sft/model /path/to/data /path/to/output

set -e

# Default values (can be overridden by command line arguments)
SFT_MODEL_PATH="${1:-/path/to/your/sft/model}"
DATA_DIR="${2:-/path/to/your/data}"
OUTPUT_DIR="${3:-./checkpoints/retool-grpo}"
EXPERIMENT_NAME="${4:-retool-grpo-experiment}"

# Validate inputs
if [[ "$SFT_MODEL_PATH" == "/path/to/your/sft/model" ]]; then
    echo "Error: Please provide SFT model path as first argument"
    echo "Usage: $0 <SFT_MODEL_PATH> <DATA_DIR> <OUTPUT_DIR> [EXPERIMENT_NAME]"
    exit 1
fi

if [[ "$DATA_DIR" == "/path/to/your/data" ]]; then
    echo "Error: Please provide data directory as second argument"
    echo "Usage: $0 <SFT_MODEL_PATH> <DATA_DIR> <OUTPUT_DIR> [EXPERIMENT_NAME]"
    exit 1
fi

# Ensure required files exist
if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
    echo "Error: $DATA_DIR/train.parquet not found"
    exit 1
fi

if [[ ! -f "$DATA_DIR/val.parquet" ]]; then
    echo "Error: $DATA_DIR/val.parquet not found"
    exit 1
fi

echo "Starting ReTool GRPO Training..."
echo "SFT Model: $SFT_MODEL_PATH"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Experiment Name: $EXPERIMENT_NAME"

# System optimization
ulimit -n 65535

# Configuration
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/retool/config"

# Core training parameters
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='retool_grpo_aligned' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    actor_rollout_ref.model.path="$SFT_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=256 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=strict \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/recipe/retool/config/tool_config/retool.yaml" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='retool_grpo' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_training_steps=100 \
    trainer.val_before_train=false \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.total_epochs=1 "$@"

echo "Training completed! Checkpoints saved to: $OUTPUT_DIR"
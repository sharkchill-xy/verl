#!/bin/bash

# Qwen2.5-Coder Python Interpreter RL Training Script
# 目标：训练Qwen2.5-Coder-Instruct模型学会使用Python解释器

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/python_interpreter_training/config"

# 编程数据集路径 - 只使用LeetCode数据集（新格式，支持<python_interpreter>标签）
code_train_path=/data/coding_leetcode_v2/train.parquet
code_test_path=/data/coding_leetcode_v2/test.parquet

train_files="['$code_train_path']"
test_files="['$code_test_path']"

echo "开始训练Qwen2.5-Coder模型学习使用Python解释器（LeetCode数据集）..."
echo "训练数据: $train_files (约1,890个样本)"
echo "测试数据: $test_files (约150个样本)"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='qwen25_coder_python_interpreter' \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/python_interpreter_training/config/tool_config/python_interpreter_tool_config.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='qwen_coder_python_interpreter' \
    trainer.experiment_name='qwen2.5-coder-1.5b_leetcode_function_call_v1' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=16 \
    trainer.val_before_train=False $@

echo "训练完成！"
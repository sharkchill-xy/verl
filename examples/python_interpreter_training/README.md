# Qwen2.5-Coder Python Interpreter RL Training

## 目标

本训练配置旨在让Qwen2.5-Coder-Instruct模型学会有效使用Python解释器来解决编程问题。这是验证stateful execution对LLM编程能力帮助的第一步实验。

## 实验设计

### 模型
- **基础模型**: Qwen/Qwen2.5-Coder-1.5B-Instruct
- **训练算法**: GRPO (Group Relative Policy Optimization)
- **多轮对话**: 支持最多10轮交互

### 工具配置
- **Python解释器**: 使用SandboxFusionTool提供安全的代码执行环境
- **执行超时**: 30秒
- **并发限制**: 10个并发工作节点

### 训练参数
- **学习率**: 1e-6
- **批次大小**: 128
- **序列长度**: 2048 (prompt + response)
- **采样数**: 每个prompt生成4个响应
- **KL散度系数**: 0.001
- **熵系数**: 0.01 (鼓励探索)

## 使用方法

### 1. 准备数据

首先准备编程相关的数据集：

```bash
# 方案1: 只使用KodCode数据集
python3 examples/python_interpreter_training/prepare_coding_data.py --output_dir /data2/lixy/coding

# 方案2: 使用多数据源（推荐）- KodCode + LeetCode
python3 examples/python_interpreter_training/prepare_coding_data_multi.py --output_dir /data2/lixy/coding_multi

# 多数据源方案包含约11,000个编程问题：
# - KodCode-Light-RL-10K: 专门为RL设计的编程问题
# - LeetCodeDataset v2: 高质量的算法和数据结构问题
```

### 2. 配置环境

确保Docker容器正在运行，并且有足够的GPU资源：

```bash
# 检查Docker容器
docker ps

# 进入VERL容器
docker exec -it verl bash
```

### 3. 运行训练

```bash
cd /home/lixy/workspace/verl

# 运行训练脚本
bash examples/python_interpreter_training/run_qwen25_coder_python_interpreter.sh
```

### 4. 监控训练

训练过程会通过Wandb进行监控，项目名称为 `qwen_coder_python_interpreter`。

## 关键特性

### 多轮交互
- 模型可以通过多轮对话逐步解决复杂问题
- 每轮可以执行Python代码并获得反馈
- 支持迭代式问题解决方法

### 工具使用学习
- 模型学会何时调用Python解释器
- 学会如何构造有效的Python代码
- 学会如何解释和使用执行结果

### 强化学习优化
- 基于任务完成情况给予奖励
- 鼓励模型探索不同的解决方案
- 逐步提高工具使用的效率和准确性

## 预期成果

训练完成后，模型应该能够：

1. **识别需要计算的问题** - 自动判断何时需要使用Python解释器
2. **生成正确的Python代码** - 针对具体问题编写合适的代码
3. **多步骤问题解决** - 通过多轮交互逐步解决复杂问题
4. **错误处理** - 根据执行结果调整代码和方法

## 后续实验

这是验证stateful execution效果的第一步。后续可以：

1. **对比实验** - 与不使用工具的版本对比
2. **更复杂任务** - 测试数据分析、算法实现等任务
3. **状态保持** - 实现真正的stateful execution并对比效果
4. **评估指标** - 设计更全面的评估方法

## 当前进展状态 (2025-06-18)

### ✅ 已完成的工作

1. **数据集准备完成（敏捷迭代 + Function Call版本）**
   - 训练数据: `/data/coding_leetcode_v2/train.parquet` (1,890个样本)
   - 测试数据: `/data/coding_leetcode_v2/test.parquet` (150个样本)
   - 数据来源: LeetCodeDataset v2 (仅LeetCode数据)
   - 数据格式: 标准OpenAI function call格式，调用`python_interpreter`函数
   - 选择理由: 采用敏捷迭代方式，使用通用function call格式便于迁移

2. **环境配置完成**
   - SandboxFusion服务运行正常: `http://210.28.135.36:8080`
   - verl容器运行正常: `docker ps` 显示容器 `verl` 状态为 Up
   - 工具配置文件已修复: `sandbox_fusion_url` 指向正确的服务地址

3. **训练框架搭建完成**
   - 主配置: `config/qwen25_coder_python_interpreter.yaml`
   - 工具配置: `config/tool_config/python_interpreter_tool_config.yaml`
   - 训练脚本: `run_qwen25_coder_python_interpreter.sh`

### 🔄 当前状态

项目已准备就绪，可以开始RL训练。需要注意的关键点：

1. **容器路径映射**
   - 宿主机项目路径: `/home/lixy/workspace/VerlCoder/verl`
   - 容器内项目路径: `/workspace/verl`
   - 数据需要拷贝到容器: 数据目录未挂载到容器中

2. **服务配置**
   - SandboxFusion API: `http://210.28.135.36:8080/run_code`
   - 测试命令: `curl 'http://210.28.135.36:8080/run_code' -H 'Content-Type: application/json' --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'`

3. **训练启动步骤**
   ```bash
   # 数据已通过容器挂载自动可用，直接启动训练
   docker exec -it verl bash
   cd /workspace/verl
   bash examples/python_interpreter_training/run_qwen25_coder_python_interpreter.sh
   ```

### 🎯 科研目标

通过 gdb-like stateful debugger 来帮助 LLM 解决竞赛难题：

1. **第一步（当前）**: 训练Qwen2.5-Coder-1.5B-Instruct学会使用Python解释器
2. **后续步骤**: 实现真正的stateful调试功能，对比效果

### 📋 下次启动时的检查清单

1. 确认SandboxFusion服务运行: `docker ps | grep sandbox`
2. 确认verl容器运行: `docker ps | grep verl`
3. 确认数据在容器中: `docker exec verl ls -la /data/coding_leetcode_v2/`
4. 检查工具配置: `grep sandbox_fusion_url /home/lixy/workspace/VerlCoder/verl/examples/python_interpreter_training/config/tool_config/python_interpreter_tool_config.yaml`

## 配置文件说明

- `config/qwen25_coder_python_interpreter.yaml` - 主要训练配置
- `config/tool_config/python_interpreter_tool_config.yaml` - Python解释器工具配置
- `run_qwen25_coder_python_interpreter.sh` - 训练脚本
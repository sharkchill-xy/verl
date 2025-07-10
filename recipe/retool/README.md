# ReTool GRPO Training Recipe

This recipe implements ReTool (Retrieval-Tool) training using GRPO (Generalized Reward Preference Optimization) for mathematical reasoning with code interpreter capabilities.

## Overview

ReTool enhances language models' mathematical reasoning by teaching them to use code interpreters effectively. This implementation uses the VERL framework with SGLang rollout backend for efficient multi-turn tool interactions.

### Key Features

- **Multi-turn tool interactions**: Up to 8 assistant/user turns with code execution
- **GRPO optimization**: Advanced reward-based preference learning
- **Sandbox integration**: Safe code execution via Sandbox Fusion API
- **Distributed training**: FSDP2 strategy with sequence parallelism support
- **Comprehensive evaluation**: AIME24 dataset evaluation framework

## Quick Start

### Prerequisites

1. **VERL Framework**: Ensure VERL is properly installed and configured
2. **SGLang Backend**: SGLang server for efficient rollout generation  
3. **Sandbox Fusion**: Code execution environment
4. **Virtual Environment**: Python environment with required dependencies

### Basic Usage

1. **Prepare your data**:
   ```bash
   ./preprocess_data.sh /path/to/input.parquet /path/to/output.parquet /path/to/tokenizer
   ```

2. **Train the model**:
   ```bash
   ./train_retool_grpo.sh /path/to/sft/model /path/to/data /path/to/output
   ```

3. **Evaluate the model**:
   ```bash
   ./evaluate_model.sh /path/to/model /path/to/results.json http://api:8000/v1 http://sandbox:8080
   ```

## Detailed Usage

### Data Preprocessing

The preprocessing script filters training data to remove overly long sequences:

```bash
./preprocess_data.sh [INPUT_FILE] [OUTPUT_FILE] [TOKENIZER_PATH]
```

**Arguments:**
- `INPUT_FILE`: Path to input parquet file
- `OUTPUT_FILE`: Path to save filtered parquet file  
- `TOKENIZER_PATH`: Path to tokenizer for length calculation

**Default filtering parameters:**
- Max prompt length: 2048 tokens
- Max response length: 4096 tokens
- Parallel workers: 192

### GRPO Training

The training script launches distributed GRPO training with multi-turn tool support:

```bash
./train_retool_grpo.sh [SFT_MODEL_PATH] [DATA_DIR] [OUTPUT_DIR] [EXPERIMENT_NAME]
```

**Arguments:**
- `SFT_MODEL_PATH`: Path to SFT checkpoint (starting point)
- `DATA_DIR`: Directory containing `train.parquet` and `val.parquet`
- `OUTPUT_DIR`: Directory for checkpoints and logs
- `EXPERIMENT_NAME`: Experiment identifier (optional)

**Key training parameters:**
- Algorithm: GRPO with low-variance KL regularization
- Batch size: 256 (configurable)
- Learning rate: 1e-6 (actor)
- Max turns: 8 assistant + 8 user
- Strategy: FSDP2 with gradient checkpointing

### Model Evaluation

The evaluation script runs AIME24 assessment with multi-sampling:

```bash
./evaluate_model.sh [MODEL_PATH] [OUTPUT_PATH] [BASE_URL] [SANDBOX_URL] [API_KEY]
```

**Arguments:**
- `MODEL_PATH`: Path to model checkpoint
- `OUTPUT_PATH`: JSON file for results
- `BASE_URL`: Model API endpoint (default: localhost:8000/v1)
- `SANDBOX_URL`: Sandbox API endpoint (default: localhost:8080)
- `API_KEY`: API authentication key (default: "EMPTY")

**Evaluation parameters:**
- Dataset: AIME24 (30 problems)
- Samples per problem: 32
- Max turns: 8
- Temperature: 0.7
- Concurrent requests: 64

## Configuration

### Tool Configuration

Tools are configured via `config/tool_config/retool.yaml`:

```yaml
tools:
  - name: "code_interpreter"
    type: "SandboxFusionTool"
    config:
      base_url: "http://your-sandbox-api:8080"
      timeout: 30
```

### Training Configuration

Main configuration template in `config/retool_grpo_template.yaml`:

- Uses Hydra variable substitution for paths
- Override via command line arguments
- Supports both small-scale testing and full training

### Distributed Setup

**FSDP2 Configuration:**
- Parameter sharding with optional offloading
- Gradient checkpointing for memory efficiency
- Sequence parallelism (Ulysses) for long sequences

**Memory Optimization:**
- Dynamic batch sizing
- Chunked entropy computation
- Gradient accumulation

## Performance Results

### Benchmark Performance (AIME24)

| Model | Pass@1 | Pass@32 | Training Method |
|-------|--------|---------|----------------|
| Qwen2.5-7B-Instruct (baseline) | 12.71% | 30.00% | - |
| ReTool SFT | 16.04% | 50.00% | Supervised Fine-tuning |
| **ReTool GRPO** | **27.50%** | **66.67%** | **GRPO Training** |

### Key Improvements

- **Pass@1 improvement**: +14.79% absolute (117% relative increase)
- **Pass@32 improvement**: +36.67% absolute (122% relative increase)
- **Tool utilization**: Effective code execution for mathematical reasoning
- **Multi-turn capability**: Leverages extended interactions for complex problems

## Advanced Usage

### Custom Data Preparation

For custom datasets, ensure your data follows the multi-turn conversation format:

```json
{
  "prompt": "Your mathematical problem",
  "response": "Solution with tool calls",
  "tools_kwargs": {...}
}
```

### Hyperparameter Tuning

Key hyperparameters to adjust:

**Learning rates:**
- Actor LR: 1e-6 (start conservatively)
- KL coefficient: 0.001 (balance exploration/exploitation)

**Batch configuration:**
- Train batch size: 256-512 (memory dependent)
- Micro batch size: 4-8 per GPU
- Gradient accumulation: Automatic via dynamic batching

**Sequence lengths:**
- Prompt length: 2048-8192 tokens
- Response length: 4096-8192 tokens
- Consider memory constraints

### Multi-Node Training

For larger scale training:

```bash
# Modify trainer config
trainer.nnodes: 4
trainer.n_gpus_per_node: 8

# Ensure proper NCCL configuration
export NCCL_TIMEOUT=600
```

### Monitoring and Debugging

**Logging integration:**
- Console output for immediate feedback
- SwanLab for experiment tracking
- Checkpoint saving every 20 steps

**Debug tips:**
- Start with small batch sizes
- Monitor GPU memory utilization
- Check sequence length distributions
- Validate tool API connectivity

## Troubleshooting

### Common Issues

**Memory errors:**
- Reduce batch size or sequence length
- Enable gradient checkpointing
- Consider parameter offloading

**Timeout errors:**
- Increase NCCL timeout settings
- Check network connectivity between nodes
- Verify sandbox API availability

**Tool call failures:**
- Validate sandbox endpoint configuration
- Check API authentication
- Monitor concurrent request limits

**Training instability:**
- Lower learning rates
- Adjust KL regularization
- Ensure data quality

### Performance Optimization

**For faster training:**
- Pre-filter datasets to avoid runtime filtering
- Use larger micro batch sizes if memory allows
- Enable mixed precision training
- Optimize data loading with more workers

**For better results:**
- Increase training steps for convergence
- Tune sampling temperature for evaluation
- Experiment with different tool configurations
- Consider curriculum learning approaches

## File Structure

```
recipe/retool/
├── README.md                           # This documentation
├── train_retool_grpo.sh               # Main training script
├── preprocess_data.sh                 # Data preprocessing
├── evaluate_model.sh                  # Model evaluation
├── config/
│   ├── retool_grpo_template.yaml      # Configuration template
│   └── tool_config/
│       └── retool.yaml                # Tool configuration
├── eval_aime24_standalone.py          # Evaluation implementation
├── prefilter_dataset.py               # Data filtering utility
└── retool.py                          # Custom tool implementations
```

## Contributing

When contributing to this recipe:

1. Follow the existing code structure and naming conventions
2. Test with small-scale runs before full training
3. Update documentation for new features
4. Ensure backward compatibility with existing configurations

## Citation

If you use this recipe in your research, please cite:

```bibtex
@article{retool2024,
  title={ReTool: Learning to Use Tools for Mathematical Reasoning},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This recipe is part of the VERL framework and follows the same licensing terms.
# PCVR HeteroFormer

A deep learning framework for **Predicted Conversion Rate (PCVR)** estimation in online advertising, built for large-scale click-through and conversion prediction tasks. This project implements a hybrid transformer architecture with explicit cross-field interactions, rotary positional embeddings, and curriculum learning strategies.

## Overview

This codebase provides an end-to-end solution for training and inference of a heterogeneous transformer model designed for advertising CTR/CVR prediction. It supports multi-domain sequential features, sparse ID embeddings, dense numerical features, and advanced training techniques including adaptive focal loss, sharpness-aware minimization (SAM), and anti-leak temporal validation splits.

## Key Features

### Architecture
- **Heterogeneous Feature Tokenization**: Separate processing pipelines for user/item sparse IDs, dense features, and variable-length sequential behaviors across multiple domains.
- **RoPE (Rotary Position Embedding)**: Replaces learnable positional embeddings with rotary frequency-based encoding for better length generalization.
- **SwiGLU Activation**: Uses SwiGLU feed-forward networks instead of standard ReLU/GeLU for improved representational capacity.
- **Top-K Sequence Compression**: Optional cross-attention-based compression for long sequences to reduce quadratic attention complexity.
- **Explicit Cross-Field Interaction**: `CrossFieldLayer` models second-order interactions (user×item, user×sequence, item×sequence) with gated residual fusion.
- **Theme-Aware Cross-Domain Attention**: `ThemeCrossDomainLayer` clusters sequence tokens into latent themes and performs cross-domain self-attention within each theme.
- **Type-Aware Attention**: Query/key modulation via learnable type embeddings to break attention symmetry across token types (CLS, user, item, context, sequence).
- **Early Exit Heads**: Intermediate layer supervision for gradient flow and uncertainty estimation.

### Training
- **Adaptive Focal Loss**: Automatically adjusts focal gamma via gradient-based meta-learning, with per-step EMA smoothing and warmup scheduling.
- **SAM Optimizer**: Sharpness-Aware Minimization wrapper for improved generalization on the dense parameter groups.
- **Isolated Parameter Groups**: Separate optimizers for sparse embeddings (Adagrad) and dense parameters (AdamW), with independent learning rates and gradient clipping.
- **Curriculum Learning**: Progressive sequence length scheduling and label smoothing annealing across epochs.
- **OHEM (Online Hard Example Mining)**: Optional hard-example-focused training by reweighting high-loss samples.
- **Stochastic Depth**: Layer-wise drop-path regularization for deep backbone training.
- **Anti-Leak Validation**: Temporal or user-level train/validation splits to prevent information leakage in sequential data.

### Data & Inference
- **Parquet Format**: Native support for Apache Parquet with efficient row-group iteration and dynamic buffer pooling.
- **Time Bucket Encoding**: Log-scale temporal discretization with precomputed lookup tables for fast feature engineering.
- **Temporal Decay Weights**: Short-term (hourly) and long-term (daily) exponential decay for sequence positions.
- **Torch.Compile Support**: Optional `torch.compile` integration for both training and inference with error suppression.
- **CUDA Graph Warmup**: Inference-time GPU warmup and TF32 acceleration for optimal throughput.

## Repository Structure

```
.
├── run.sh              # Training launcher with environment configuration
├── train.py            # Training entry point: argument parsing, model construction, trainer initialization
├── trainer.py          # Core trainer: loss computation, optimization loop, evaluation, checkpointing
├── model.py            # Model architecture: tokenizers, encoders, backbone, heads, and PCVRHeteroFormer
├── dataset.py          # Data pipeline: Parquet reader, feature schema, curriculum scheduling, anti-leak splits
└── infer.py            # Inference script: model loading, batch processing, prediction export
```

## Quick Start

### Prerequisites

- Python >= 3.9
- PyTorch >= 2.0 with CUDA support
- PyArrow & Pandas
- scikit-learn, tqdm

Install dependencies:
```bash
pip install torch pyarrow pandas numpy scikit-learn tqdm
```

### Training

Configure paths and hyperparameters via environment variables or command-line arguments:

```bash
export TRAIN_DATA_PATH=/path/to/train/parquets
export TRAIN_CKPT_PATH=./checkpoints
export TRAIN_LOG_PATH=./logs
export TRAIN_TF_EVENTS_PATH=./tensorboard

bash run.sh
```

Key training arguments (see `train.py` for full list):
- `--data_dir`: Directory containing `.parquet` files and `schema.json`
- `--batch_size`: Training batch size (default: 256)
- `--d_model`: Hidden dimension (default: 128)
- `--num_layers`: Backbone transformer layers (default: 4)
- `--num_heads`: Attention heads (default: 4)
- `--lr`: Dense parameter learning rate (default: 1e-4)
- `--sparse_lr`: Sparse embedding learning rate (default: 0.05)
- `--loss_type`: `focal` or `bce`
- `--use_sam`: Enable SAM optimizer
- `--seq_top_k`: Enable Top-K sequence compression (0 to disable)
- `--rope_base`: RoPE base frequency (default: 10000.0)

### Inference

```bash
export MODEL_OUTPUT_PATH=./checkpoints/global_stepXXXX.best_model
export EVAL_DATA_PATH=/path/to/test/parquets
export EVAL_RESULT_PATH=./results

python infer.py
```

The inference script will:
1. Load `train_config.json` and `model.pt` from the checkpoint directory
2. Reconstruct the model architecture with exact training hyperparameters
3. Process test Parquet files and output `predictions.json`

## Data Format

The framework expects Apache Parquet files with the following columns:

**Metadata**
- `timestamp`: Event timestamp (int64)
- `label_type`: 0 (no click), 1 (click), 2 (conversion)
- `user_id`: User identifier string

**User Features**
- `user_int_feats_{fid}`: Integer/sparse user features (scalar or list)
- `user_dense_feats_{fid}`: Float/dense user features (scalar or list)

**Item Features**
- `item_int_feats_{fid}`: Integer/sparse item features

**Sequential Features** (per domain, e.g., `seq_a`, `seq_b`)
- `{prefix}_{fid}`: Historical behavior features (list arrays)
- `{prefix}_{ts_fid}`: Historical timestamps (list arrays)

A `schema.json` file must define the feature mapping:
```json
{
  "user_int": [[fid, vocab_size, dim], ...],
  "item_int": [[fid, vocab_size, dim], ...],
  "user_dense": [[fid, dim], ...],
  "seq": {
    "seq_a": {
      "prefix": "seq_a",
      "ts_fid": 999,
      "features": [[fid, vocab_size], ...]
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TRAIN_DATA_PATH` | Training data directory |
| `TRAIN_CKPT_PATH` | Checkpoint save directory |
| `TRAIN_LOG_PATH` | Log output directory |
| `TRAIN_TF_EVENTS_PATH` | TensorBoard event directory |
| `ANTI_LEAK_MODE` | Validation split mode: `timestamp`, `user_id`, or `none` |
| `CURRICULUM_MAX_LEN` | JSON dict for per-domain sequence length limits |
| `SEQ_TOP_K` | Override Top-K compression setting |
| `TORCH_COMPILE_DISABLE` | Set to `1` to disable torch.compile |

### Model Hyperparameters

The model supports extensive architectural customization:
- Embedding dimensions and vocabulary thresholds
- Transformer depth, width, and attention configuration
- Stochastic depth probability and dropout rates
- Cross-field interaction interval and theme counts
- Sequence encoder layers and compression settings

All hyperparameters are persisted in `train_config.json` during training and automatically resolved during inference.

## Advanced Topics

### Anti-Leak Data Splits

To prevent temporal leakage in sequential recommendation:
- **Timestamp mode**: Sorts row groups by time and splits temporally.
- **User mode**: Performs user-level train/validation splits.
- **Vocab clamping**: Inference-time OOB vocab values are clamped to training-set maximums.

### Curriculum Learning

Sequence lengths are progressively increased over the first few epochs via `CurriculumScheduler`. This can be controlled via:
- Environment variable `CURRICULUM_MAX_LEN`
- Trainer argument `curriculum_epochs`
- Dataset method `set_curriculum_max_len()`

### Multi-Process Data Loading

The `PCVRParquetDataset` implements `IterableDataset` with:
- Worker-specific row group sharding
- Dynamic buffer pooling for zero-copy batch assembly
- Shuffle buffering with configurable `buffer_batches`

## License

This project was developed for the Tencent Advertising Algorithm Competition. All rights reserved by the author.

## Acknowledgments

- PyTorch team for `torch.compile` and scaled dot-product attention
- HyFormer and MoS architecture inspirations for cross-domain interaction designs

#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export TORCHINDUCTOR_CPP_WRAPPER=0
export TORCHINDUCTOR_CPP_BUILDER=0
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NVIDIA_TF32_OVERRIDE=1
export TORCH_COMPILE_DISABLE=1
export ANTI_LEAK_MODE=timestamp

# v11.2 curriculum: 初始环境变量（trainer会动态覆盖）
export CURRICULUM_MAX_LEN='{"seq_a":50,"seq_b":50,"seq_c":50,"seq_d":50}'

# v11.2: Optional sequence compression for long sequences (>1k)
# Set to 0 to disable, or 50/100 for aggressive compression
export SEQ_TOP_K=0

python3 --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

rm -rf /tmp/torchinductor_$(whoami)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

train_args=(
    --data_dir "$TRAIN_DATA_PATH"
    --ckpt_dir "$TRAIN_CKPT_PATH"
    --log_dir "$TRAIN_LOG_PATH"

    # Data & System
    --num_workers 8
    --batch_size 256
    --buffer_batches 50
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"

    # Model Architecture (v11.2 Hybrid)
    --d_model 128
    --emb_dim 16
    --num_layers 4
    --num_heads 4
    --dropout_rate 0.15
    --stochastic_depth_prob 0.1
    --id_vocab_threshold 10000
    --emb_skip_threshold 10000000

    # v11.2 NEW: Sequence compression and RoPE
    --seq_top_k "${SEQ_TOP_K:-0}"
    --rope_base 10000.0

    # Training
    --lr 1e-4
    --sparse_lr 0.05
    --num_epochs 999
    --patience 5
    --warmup_steps 300
    --eval_every_n_steps 600

    # Loss (v11 Adaptive Focal)
    --loss_type focal
    --focal_alpha_pos 0.6
    --focal_alpha_neg 0.4
    --focal_max_gamma 4.0

    # v11.2 Training Strategies
    --use_sam
    --sam_rho 0.05
    --ohem_ratio 1.0
    --curriculum_epochs 3
    --label_smoothing_max 0.05
    --label_smoothing_min 0.001
    --early_exit_weight 0.3

    # torch.compile (disabled by default for stability)
    # --compile

    "$@"
)

python3 -u "${SCRIPT_DIR}/train.py" "${train_args[@]}"
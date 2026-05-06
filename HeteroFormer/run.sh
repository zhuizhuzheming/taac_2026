#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# v7.1: Allow scalar outputs but safer compile settings
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export TORCHINDUCTOR_CPP_WRAPPER=0
export TORCHINDUCTOR_CPP_BUILDER=0
export TORCHINDUCTOR_MAX_AUTOTUNE=0
# v7.1: CUDA optimizations
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NVIDIA_TF32_OVERRIDE=1

python3 --version

#pip install --upgrade pip
# 1. 卸载当前 PyTorch（谨慎操作，确保环境正确）
#pip uninstall -y torch torchvision torchaudio

# 2. 安装 PyTorch 2.5.1 (CUDA 12.1)
#pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -f https://mirrors.aliyun.com/pytorch
#-wheels/cu118.html -i https://mirrors.aliyun.com/pypi/simple/

# 3. 验证安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 1. 删除 Inductor 编译缓存（非常重要！）
rm -rf /tmp/torchinductor_$(whoami)

# 2. 清除项目中可能的 __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# ---- v7.1 Architecture-Preserving GSE Edition ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_groups_json "" \
    --emb_skip_threshold 10000000 \
    --num_workers 8 \
    --data_dir $TRAIN_DATA_PATH \
    --ckpt_dir $TRAIN_CKPT_PATH \
    --log_dir $TRAIN_LOG_PATH \
    --batch_size 256 \
    --lr 1e-4 \
    --sparse_lr 0.001 \
    --warmup_steps 1000 \
    --num_epochs 999 \
    --patience 5 \
    --buffer_batches 50 \
    --d_model 128 \
    --emb_dim 16 \
    --num_layers 4 \
    --rank_schedule bottleneck \
    --dropout_rate 0.2 \
    --loss_type focal \
    --focal_alpha 0.1 \
    --focal_gamma 2.0 \
    --use_grafted_optimizer \
    --gate_anneal_steps 3000 \
    --stochastic_depth_prob 0.2 \
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    --label_smoothing_strategy none \
    --label_smoothing_max_eps 0.08 \
    --label_smoothing_min_eps 0.001 \
    --label_smoothing_anneal_steps 2000 \
    --sparse_weight_decay 1e-4 \
    --id_dropout_rate 0.0 \
    --seq_id_dropout_rate 0.0 \
    --id_vocab_threshold 10000 \
    --shrinkage 0.05 \
    --base_rank 64 \
    --gse_num_codes 64 \
    --gse_code_dim 64 \
    --gse_num_layers 4 \
    --gse_aux_weight 0.3 \
    --focal_alpha_pos 0.6 \
    --focal_alpha_neg 0.4 \
    --focal_max_gamma 4.0 \
    --compile_suppress_errors\
    --compile_backend aot_eager \
    --compile_mode default \
    --compile_dynamic \
    --prior_weight 0.02 \
    --ece_weight 0.02 \
    --lambdarank_weight 0.1 \
    --global_ctr 0.01 \
    --zmlc_lambda 0.0 \
    --cross_network_layers 2 \
    --label_smoothing_max_eps 0.0 \
    --label_smoothing_min_eps 0.0 \
    "$@"
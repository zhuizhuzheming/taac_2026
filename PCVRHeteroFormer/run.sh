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
python3 --version

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

rm -rf /tmp/torchinductor_$(whoami)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

train_args=(
    --ns_groups_json ""
    --emb_skip_threshold 10000000
    --num_workers 8
    --data_dir "$TRAIN_DATA_PATH"
    --ckpt_dir "$TRAIN_CKPT_PATH"
    --log_dir "$TRAIN_LOG_PATH"
    --batch_size 256
    --lr 1e-4
    --sparse_lr 0.001
    --warmup_steps 300          # 与 MetaAligner 内部 warmup 对齐
    --num_epochs 999
    --patience 5
    --buffer_batches 50
    --d_model 128
    --emb_dim 16
    --num_layers 4
    --rank_schedule bottleneck
    --loss_type focal
    --focal_alpha 0.1
    --focal_gamma 2.0
    --stochastic_depth_prob 0.1
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
    --label_smoothing_strategy hybrid
    --label_smoothing_max_eps 0.05
    --label_smoothing_min_eps 0.001
    --label_smoothing_anneal_steps 2000
    --sparse_weight_decay 1e-4
    --id_vocab_threshold 10000
    --shrinkage 0.05
    --base_rank 64
    --cross_network_layers 2
    --focal_alpha_pos 0.6
    --focal_alpha_neg 0.4
    --focal_max_gamma 4.0
    --global_ctr 0.01
    --compile_suppress_errors
    --compile_backend aot_eager
    --compile_mode default
    --no-compile_dynamic
    --num_codes 128
    --sinkhorn_iter 20
    --min_mass_ratio 0.005
    --coherence_threshold 0.15
    --lie_rank 8
    --dropout_rate 0.2          # 全局dropout
    --id_dropout_rate 0.05       # 特征ID随机丢弃
    --seq_id_dropout_rate 0.05   # 序列ID随机丢弃

    # v10 关键参数
    --kappa_base 2.0
    --sinkhorn_epsilon 0.05
    --packing_weight 0.01

    # 【v10 新增】生成式语义层自监督权重
    --ib_weight 0.0             # 信息瓶颈约束
    --recon_weight 0.0          # 重建质量
    --ortho_weight 0.0          # 正交约束

    # 【v10 保留】Energy 校准器参数
    --energy_margin 1.0
    --energy_weight 0.0

    # 【v10 移除】以下参数已废弃，由架构内部自动管理
    # --use_diffusion              # 已移除：DiffusionExplainer 内联
    # --use_energy                 # 已移除：EnergyCalibrator 内联
    # --use_domain_adversarial     # 已移除：v10 不再使用
    # --diff_weight 0.05           # 已移除：由 recon_weight 替代
    # --energy_weight 0.1          # 已保留：energy ranking loss 权重
    # --meta_update_interval 100   # 已移除：MetaAligner 简化
    # --curriculum_warmup 5000     # 已移除：v10 不再使用
    # --diffusion_warmup 1000      # 已移除：v10 不再使用

    # 验证频率
    --eval_every_n_steps 600

    "$@"
)

python3 -u "${SCRIPT_DIR}/train.py" "${train_args[@]}"
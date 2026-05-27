# HeteroFormer

**Tightly Coupling Sequence Prototypes with Heterogeneous Feature Interaction for CTR Prediction**

2026TAAC-team_id: ICPC和AI谁强

> **Note:** This repository is currently under preparation for public release. The full paper is available in [`technical_report.pdf`](./technical_report.pdf).

---
## Overview

HeteroFormer is a unified architecture for Click-Through Rate (CTR) prediction that fundamentally rethinks how long-range user behavior sequences interact with heterogeneous non-sequential features. Rather than following the prevailing two-stage paradigm—where sequences are compressed into fixed-length vectors before being fused with dense features—HeteroFormer introduces a single principle:

> **Sequence-derived semantics should actively parameterize heterogeneous feature interactions, rather than being passively injected as compressed context.**

This is achieved through a **Dynamic Prototype Manifold**—a set of learnable semantic anchors whose geometry is conditioned on user features via Cayley rotation and whose assignment is computed through entropy-regularized optimal transport (Langevin-Sinkhorn). These prototype assignments serve as soft biases and gating signals within a cross-field interaction network, enabling user-specific sequential patterns to directly shape attention topologies and feed-forward transformations.

### Key Innovations

- **Dynamic Prototype Manifold**: Maps behavior sequences onto a user-conditioned manifold via Cayley rotation and Langevin-Sinkhorn optimal transport, yielding sparse, differentiable, and interpretable prototype assignments.
- **Proto-Conditioned Cross-Field Interaction**: Prototype assignments bias multi-head cross-field attention and gate FiLM-conditioned feed-forward networks, tightly coupling sequential semantics with heterogeneous feature interaction.
- **Generative Semantic Layer**: Comprises a Diffusion Explainer (information-bottleneck residual encoder) and an Energy Calibrator (error-predictive scoring function) that capture complementary residual signals without destabilizing the primary CTR objective.
- **Decoupled Semantic Optimization (DSO)**: A training methodology that isolates generative and discriminative objectives through gradient detachment and governs their coupling via an overfitting-aware MetaAligner.
- **Semantic IDs for Interpretability**: The generative pathway naturally induces discrete, interpretable indices over the prototype manifold, replacing opaque attention weights with semantically grounded mixture coefficients.

---

## Repository Structure

This repository contains two implementations of the HeteroFormer architecture, reflecting different stages of the research-to-production pipeline:

```
.
├── HeteroFormer_model/          # Original research implementation (Full HeteroFormer)
│   └── trainer.py               # DSO trainer with MetaAligner
├── PCVRHeteroFormer/            # Industrial-strength stable variant (Stability Ablation)
│   └── ...                      # Production-ready components
├── technical_report.pdf         # Full technical report
└── README.md                    # This file
```

### `HeteroFormer_model/` — Full Architecture

The original research implementation as described in the technical report (§3–§4). This version includes:

- **SSM Cell** with continuous-time discretization for sequence encoding
- **Cayley Rotation** for user-conditioned prototype geometry
- **Langevin-Sinkhorn** optimal transport for sparse prototype assignment
- **Full DSO pipeline** with gradient isolation and MetaAligner
- **Diffusion Explainer** with orthogonal residual projection
- **Energy Calibrator** with energy-target supervision

**Characteristics**: Mechanistically validated on the Tencent Ads Algorithm Competition dataset. Achieves validation AUC peak of **0.8383** with zero NaN recoveries under joint multi-objective training. Best suited for research reproduction and ablation studies.

### `PCVRHeteroFormer/` — Stability-Abled Industrial Variant

A surgically simplified variant engineered for deployment under single-GPU, mixed-precision constraints (§5 of the technical report). Substitutions include:

| Component | Full (`HeteroFormer_model/`) | Ablated (`PCVRHeteroFormer/`) |
|-----------|------------------------------|-------------------------------|
| Sequence Encoder | SSM with continuous-time encoding | RoPE-based Transformer |
| Prototype Layer | Cayley + Sinkhorn (O(K²)) | Soft theme routing |
| Training | DSO + MetaAligner | Joint loss optimization |
| Calibration | Diffusion + Energy | Softplus MLP head |

**Characteristics**: Validation AUC improves monotonically from 0.78 to 0.83, confirming that the core principle of sequence-parameterized interaction remains effective under severe resource constraints. The simplified uncertainty module trades some discriminative calibration power for training stability and memory efficiency. Recommended for industrial deployment.

---

## Architecture

### Why Not Standard Transformer?

Standard self-attention is structurally mismatched to heterogeneous CTR features for three reasons:

1. **Feature Heterogeneity Violates Token Homogeneity**: User IDs, item attributes, context features, and sequences live on different statistical manifolds. Standard attention assumes a shared Euclidean metric, inducing distorted similarity measures.
2. **Permutation Invariance is a Liability**: In CTR, field order carries strong semantic meaning ("who → what → when"). A permutation-invariant mixer destroys this causal structure.
3. **Quadratic Complexity is Prohibitive**: With thousands of tokens in industrial settings, O(M²d) attention is infeasible. HeteroFormer treats each field as a single token (M=4), reducing attention to O(16d).

### Core Design

The architecture consists of four tightly integrated stages:

1. **Multi-View Encoding**: User, item, and dense features are encoded into shared semantic representations with task-specific views (CTR, diffusion, energy).
2. **Continuous Sequence Modeling**: Variable-length behavior sequences are processed by sequence encoders, producing multi-scale summaries (short-term, long-term, static).
3. **Dynamic Prototype Manifold**: The fused sequence representation is mapped onto a user-conditioned prototype manifold, yielding sparse prototype assignments π ∈ Δ^(K−1) and a compact prototype representation.
4. **Proto-Conditioned Interaction & Prediction**: Prototype assignments bias cross-field attention and gate feed-forward transformations; a calibrated prediction head fuses static, prototype, and residual signals.

### Decoupled Semantic Optimization (DSO)

Training HeteroFormer requires balancing three competing objectives: CTR classification, generative semantic losses, and geometric regularization. DSO achieves stable co-training through:

- **Gradient Isolation**: Architectural detachment ensures generative gradients do not backpropagate through the CTR head, and vice versa.
- **MetaAligner**: An overfitting-aware controller that dynamically modulates the auxiliary coupling weight λ_aux based on residual benefit and train-validation AUC gap.
- **Isolated Parameter Update**: Four semantic parameter groups (shared, generative, sparse, meta) maintain independent optimizer states.

---

## Experimental Results

Results on the **TencentGR** dataset from the 2026 Tencent Ads Algorithm Competition:

### Full Implementation (`HeteroFormer_model/`)

| Split | AUC | LogLoss |
|-------|-----|---------|
| Validation (peak at step 1,861) | **0.8383** | 0.40 |
| Test (final submission) | **0.7728** | **0.34** |

**Key Observations**:
- MetaAligner transitions from fast to stable mode around step 500, suppressing λ_aux from 0.12 to 0.03–0.05 in response to overfitting pressure.
- Prototype assignment entropy stabilizes at 4.3 (89% of theoretical maximum), indicating healthy soft semantic IDs.
- Energy Calibrator learns to discriminate between low-uncertainty and high-uncertainty samples, with mean energy growing from 0.7 to 1.4.
- Zero NaN recoveries across 2,500 training steps.

### Stability Ablation (`PCVRHeteroFormer/`)

| Metric | Trend |
|--------|-------|
| Validation AUC | 0.78 → 0.83 (monotonic) |
| Validation LogLoss | 0.31 → 0.25 |

**Trade-offs**: The ablation confirms that the core principle of sequence-parameterized interaction remains trainable and effective even when the optimal-transport machinery is replaced by lightweight alternatives. However, the simplified uncertainty module suffers from:
- Uncertainty collapse (near-constant σ ≈ 0.7)
- Logit drift (mean logits descend to −3 vs. −0.4 in full implementation)
- Training volatility due to fixed λ_aux without MetaAligner's adaptive damping

---

## Mechanistic Validation

Rather than chasing leaderboard rankings, our experimental analysis focuses on verifying that every module behaves according to its intended control logic:

1. **MetaAligner as Adaptive Optimizer**: Residual benefit r̄ remains strictly positive (~0.25) even at reduced coupling strength, confirming that DSO's dynamic coupling maximizes auxiliary contribution while respecting stability constraints.
2. **Prototype Health**: Concentration κ stable at ~1.8; assignment entropy at 4.3 corresponds to an effective "active prototype count" of ~74 out of 128.
3. **Energy Calibrator**: Standard deviation expands to 0.8, indicating successful discrimination of "semantic conflict" cases where contradictory prototypes are simultaneously activated.
4. **Residual Path**: Base-vs-final logits drift to −0.4, confirming conservative correction rather than over-confidence amplification.

---

## Interpretability & Semantic IDs

HeteroFormer replaces post-hoc explainability with built-in interpretability:

- **Semantic IDs**: The assignment vector π functions as a probabilistic membership vector over K latent behavioral concepts (e.g., "price-sensitive browsing," "impulse purchasing," "research-heavy comparison").
- **Residual Counterfactuals**: Large ‖δ‖ signals deviation from established semantic profiles, triggering real-time fallbacks or offline auditing.
- **Energy as Semantic Confidence**: High energy often indicates semantic conflict—users simultaneously activating contradictory prototypes—providing actionable diagnostics beyond generic "low confidence" flags.

---

## Citation

If you find this work useful, please consider citing our technical report:

```bibtex
@article{xu2026heteroformer,
  title={HeteroFormer: Tightly Coupling Sequence Prototypes with Heterogeneous Feature Interaction for CTR Prediction},
  author={Xu Jiahao},
  email={zhuizhuzheming@163.com}
  year={2026},
  journal={Tencent Ads Algorithm Competition Technical Report}
}
```

---

## Acknowledgments

This work was developed during the **2026 Tencent Ads Algorithm Competition**. We thank the organizers for providing the industrial-scale TencentGR dataset and the competition platform that enabled extensive diagnostic logging and mechanistic validation.We also thank Kimi K2.6 to help organize the models' implementation and the written report.

---

## License

This project is licensed under the MIT License.

---

*Last updated: May 2026*

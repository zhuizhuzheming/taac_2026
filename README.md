# HeteroFormer: A Geometric-Dynamic Approach to Sequential Feature Interaction in Conversion Rate Prediction

## Abstract

We present HeteroFormer, a neural architecture that redefines the intersection of sequential modeling and feature interaction for Post-Click Conversion Rate (PCVR) prediction. Departing from the conventional paradigm of "embed-then-interact," HeteroFormer introduces a three-tier framework: (1) continuous-time state-space sequential encoding, (2) manifold-based prototype assignment with optimal transport, and (3) residual representation surgery with online uncertainty-driven task scheduling. The core insight is that user behavior sequences and cross-field feature interactions should not be treated as separate stages—concatenated vectors fed into MLPs—but as coupled dynamical systems evolving on structured geometric spaces. We detail the mathematical foundations, architectural decisions, and emergent properties of this coupling, with particular attention to how temporal dynamics inform feature geometry and how geometric constraints regularize temporal representations.

---

## 1. The Duality Problem: Sequence and Feature as Two Solitudes

### 1.1 Conventional Architectures and Their Limitations

Most deep learning models for recommendation tasks adopt a pipeline architecture:

```
Raw Features → Embedding Lookup → Feature Interaction → Sequence Encoding → Prediction Head
```

This sequential decomposition, while modular, imposes a fundamental epistemological constraint: **feature interaction and sequence modeling are treated as independent subproblems whose outputs are merely concatenated**. Consider the prevalent designs:

- **Factorization Machines (FM)** and variants (DeepFM, xDeepFM) compute feature interactions via inner products or compressed sensing operations on embedding tables. The sequence, if present, is reduced to a single vector via mean/max pooling before entering the interaction phase.
- **Transformer-based sequential models** (SASRec, BERT4Rec) excel at capturing item-item transitions but treat user profile features as static prefix tokens or side information, not as dynamical agents that co-evolve with the sequence.
- **Two-tower architectures** explicitly separate user and item towers, making cross-feature interactions within each tower impossible and cross-tower interactions limited to final dot products.

The consequence is a **representational bottleneck**: the sequence encoder produces a fixed-length vector that must simultaneously encode (a) temporal dynamics, (b) user intent drift, (c) item affinity evolution, and (d) context sensitivity—before any of these factors can interact with non-sequential features. By the time the sequence vector reaches the feature interaction module, the fine-grained temporal structure has been irreversibly compressed.

### 1.2 The Coupling Hypothesis

HeteroFormer is built on an alternative hypothesis: **the geometry of feature interaction should be conditioned on the temporal dynamics of the sequence, and the temporal evolution of the sequence should be constrained by the geometric structure of feature fields.**

This is not merely about feeding sequence representations into feature interaction modules earlier or with more connections. It is about designing architectures where:

1. **Time is a first-class differentiable quantity**, not a positional index.
2. **Feature fields are dynamical agents** that exchange information through a shared geometric substrate.
3. **Uncertainty is an explicit representational dimension**, not noise to be suppressed.
4. **The interaction between sequence and features is itself learnable and regulatable**, not hard-coded in the architecture.

The following sections unfold how these principles are operationalized.

---

## 2. Continuous-Time Sequence Encoding: Beyond Positional Indices

### 2.1 The Time-Interval Semantics Problem

In PCVR prediction, the raw time difference between consecutive user actions carries direct semantic meaning:
- A click followed by a purchase within 5 minutes suggests impulse buying.
- The same action pair separated by 3 days suggests deliberative comparison.

Standard positional encodings (sinusoidal or learned) map time to indices: position 0, 1, 2, ..., L. This encoding is **translation-invariant** (shifting the entire sequence does not change relative positions) but **not scale-invariant** (the model cannot generalize to sequences with different granularities). More critically, it is **not differentiable with respect to raw time values**—the model learns correlations between positions, not functional relationships between time intervals.

### 2.2 State Space Models as Continuous-Time Discretizations

HeteroFormer adopts the State Space Model (SSM) framework from continuous-time signal processing. The continuous dynamics are:

```
h'(t) = A · h(t) + B · x(t)
y(t)  = C · h(t) + D · x(t)
```

where h(t) ∈ R^N is the hidden state, x(t) ∈ R^D is the input signal, and A ∈ R^(N×N), B ∈ R^(N×D), C ∈ R^(D×N), D ∈ R^(D×D) are system matrices.

For irregularly sampled discrete observations at times t_0, t_1, ..., t_{L-1}, the exact discretization with step size Δ_k = t_{k+1} - t_k is:

```
h_k = Ā_k · h_{k-1} + B̄_k · x_k

where Ā_k = exp(A · Δ_k)      (matrix exponential)
      B̄_k = A^{-1}(Ā_k - I) · B   (for invertible A)
```

The matrix exponential `exp(A · Δ)` is the key operator: it propagates the hidden state forward by a continuous time interval Δ, not by a fixed step.

### 2.3 Structured State Matrix: Multi-Scale Exponential Decay

A critical design choice is the parameterization of A. Rather than learning A directly (which leads to unstable dynamics and expensive matrix exponentials), we constrain A to be diagonal with structured negative values:

```
A_{nn} = -exp(α_n)   where α_n = linspace(0, 3, state_dim)
```

This yields **exponentially decaying modes with varying time constants**:
- Mode 0: τ_0 ≈ 1/1 = 1 second (captures immediate reactions)
- Mode N-1: τ_{N-1} ≈ 1/exp(3) ≈ 0.05 seconds (captures micro-patterns)
- Intermediate modes: logarithmically spaced between these extremes

The physical interpretation is a **multi-resolution memory**: the hidden state maintains fine-grained recent history alongside coarse-grained distant history, with each dimension responsible for a specific temporal scale. This is fundamentally different from multi-head attention, where different heads attend to different positions but all positions share the same time granularity.

### 2.4 Adaptive Discretization via Learned Time Warping

Raw time differences in user behavior data span enormous ranges (seconds to months). Directly using Δ_k = t_{k+1} - t_k in seconds would cause numerical instability (exp(-1000000) underflows to zero). HeteroFormer introduces a learned warping:

```
Δ_k^{scaled} = softplus(W_Δ · log(Δ_k + ε) + b_Δ)
```

where the projection network `delta_proj` maps log-time-differences to positive step sizes, clipped to [10^{-4}, 1.0]. This achieves two goals:
1. **Logarithmic compression**: decades of time variation are compressed to a manageable range.
2. **Learned relevance**: the network learns which time scales are predictive for the task, down-weighting irrelevant intervals.

### 2.5 Parallel Scan: From Recurrence to Convolution

The naive implementation of SSM recurrence is serial: `h_k` depends on `h_{k-1}`, preventing GPU parallelization. HeteroFormer employs the **parallel scan algorithm** (also known as parallel prefix sum or Blelloch scan):

```
Given: a_k = Ā_k, b_k = B̄_k · x_k

Define associative operator ⊕:
    (a_i, b_i) ⊕ (a_j, b_j) = (a_j · a_i, a_j · b_i + b_j)

Then: (Ā_{0→k}, h_k) = (a_0, b_0) ⊕ (a_1, b_1) ⊕ ... ⊕ (a_k, b_k)
```

This reduces the computation from O(L) sequential steps to O(log L) parallel steps, making SSMs competitive with Transformers in training throughput while maintaining linear (rather than quadratic) complexity in sequence length.

### 2.6 Multi-Scale Sequence Pooling: Temporal Attention as Physics

The SSM outputs a full hidden sequence `h ∈ R^(B×L×D)`. For CTR prediction, this must be compressed to a user vector. Rather than simple mean pooling, HeteroFormer designs three pooling operators with explicit physical interpretations:

| Pooling | Weight Function | Physical Interpretation |
|---------|----------------|------------------------|
| **Short-term** | w_k ∝ exp(-k/10) · mask_k | Exponential decay with 10-step halftime; captures recency bias |
| **Long-term** | w_k ∝ mask_k / Σ mask | Uniform average over valid positions; captures stable preferences |
| **Static** | f(mean[h], std[h], last[h]) | Statistical fingerprint: central tendency, variability, and terminal state |

The **empty sequence problem** (new users with no history) is handled not by zero-padding but by learning a **parameterized prior distribution** over hidden states:

```
h_empty ~ N(μ_empty, σ_empty²)   where μ, σ are learned parameters
```

This prior is sampled during training (via reparameterization) and fixed to the mean during evaluation, providing calibrated uncertainty for cold-start users.

---

## 3. Manifold-Based Feature Interaction: The Prototype Geometry

### 3.1 From Euclidean Embedding to Riemannian Manifolds

Traditional feature interaction assumes all representations live in a flat Euclidean space R^D, with similarity measured by dot product or cosine. HeteroFormer challenges this by introducing a **learned Riemannian manifold** where:
- **Points on the manifold** are user interest prototypes.
- **Distances on the manifold** are task-specific and user-conditioned.
- **Assignments to the manifold** are probabilistic and optimal-transport-regularized.

The manifold is operationalized through the DynamicPrototypeManifold module, which performs four operations:

### 3.2 Global Prototypes as Interest Atoms

The manifold is discretized into `num_codes` prototype vectors `η ∈ R^(K×D)` where K = num_codes and D = code_dim. These prototypes are not cluster centroids (which would require pre-clustering or online k-means) but **free parameters** learned end-to-end. Each prototype represents an "interest atom"—an elementary, interpretable unit of user preference.

The prototypes are constrained to the unit hypersphere:

```
η_k = η_k / ||η_k||   (L2 normalization)
```

This normalization is crucial: it ensures that the subsequent similarity computation is purely directional (cosine-based), removing the confounding effect of vector magnitude.

### 3.3 User-Conditioned Rotation: Cayley Transform

A fixed set of prototypes cannot capture the diversity of user interests. HeteroFormer employs a **user-conditioned rotation** via the Cayley transform, which parameterizes the special orthogonal group SO(D) (rotations in D dimensions) without explicit matrix exponentials:

```
Given skew-symmetric matrix S = U·W·V^T - V·W·U^T
where U, V ∈ R^(D×R) are tall matrices (R << D is the Lie algebra rank)
      W = diag(w) and w = tanh(MLP(user_feat))

The Cayley rotation is:
    Q = (I + S/2)·(I - S/2)^{-1}

And the rotated prototypes are:
    μ_u,k = Q · η_k
```

The Cayley transform guarantees that Q is exactly orthogonal (Q^T·Q = I), preserving distances and avoiding the dimensional collapse that plagues unconstrained linear transformations. The rank-R constraint (R=8 << D=128) ensures that the rotation is low-dimensional and generalizable.

**Physical interpretation**: Different users "see" the same set of interest atoms through different "coordinate systems." A fashion enthusiast and a tech enthusiast share the same prototype vocabulary but project them onto orthogonal subspaces of their personal interest spaces.

### 3.4 von Mises-Fisher Distribution: Directional Soft-Clustering

Given a sequence encoding `z_seq` (from the SSM) and the user-rotated prototypes `μ_u`, the assignment is computed via the von Mises-Fisher (vMF) distribution on the unit sphere:

```
p(z_seq | μ_u,k, κ) ∝ exp(κ · μ_u,k^T · z_seq)
```

where κ > 0 is the concentration parameter. Unlike Gaussian mixture models, vMF:
1. Naturally respects the spherical constraint (both z_seq and μ_u are normalized).
2. Has a single scalar precision parameter κ rather than a full covariance matrix.
3. Admits closed-form maximum likelihood estimation for κ.

HeteroFormer makes κ **user-conditioned and time-decayed**:

```
κ(u, Δt) = softplus(κ_base + MLP(user_feat)) · exp(-λ · Δt/86400)
```

- **Active users** (recent behaviors, high engagement) receive high κ: their sequence is sharply assigned to a few prototypes.
- **Dormant users** (long inactivity) receive low κ: their sequence is softly distributed across many prototypes, reflecting uncertainty.
- **Temporal decay** λ ensures that stale sequences are treated with appropriate skepticism.

### 3.5 Entropy-Regularized Optimal Transport: The Sinkhorn Mechanism

The raw vMF similarities `log_probs = κ · cosine_sim` are not directly normalized. HeteroFormer treats prototype assignment as an **optimal transport problem**: each sequence sample (mass=1) must be distributed across K prototypes, minimizing the cost matrix C = -log_probs while maximizing entropy.

The solution is the Sinkhorn-Knopp iteration:

```
K = exp(-C/ε)          (Gibbs kernel with temperature ε)
u^{(0)} = v^{(0)} = 1

Repeat:
    u^{(t+1)} = 1 / (K · v^{(t)})
    v^{(t+1)} = 1 / (K^T · u^{(t+1)})

Assignment: Π = diag(u) · K · diag(v)
```

The entropy regularization ε prevents hard assignment (which would collapse gradients and lose prototype utilization diversity). HeteroFormer further adds **Langevin noise** in the final iterations:

```
u = u + randn_like(u) · σ   (during training only)
```

This stochastic perturbation acts as a **regularizer against mode collapse**—preventing all sequences from being assigned to the same dominant prototype—and as a **mechanism for exploration** in the assignment space.

### 3.6 Task-Specific Prototype Projections

The prototypes are shared across tasks, but their semantic interpretation is task-conditioned:

```
μ_u,k^{task} = TaskProj_{task}(μ_u,k)   for task ∈ {ctr, diff, energy}
```

This allows the same geometric structure (the manifold) to support different objectives:
- **CTR**: prototypes are discriminative boundaries between converters and non-converters.
- **Diffusion**: prototypes are anchor points for denoising trajectories.
- **Energy**: prototypes are basins in the energy landscape.

The final representation is the task-specific weighted average:

```
proto_repr^{task} = Σ_k Π_k · μ_u,k^{task}
```

---

## 4. Cross-Field Dynamics: Feature Interaction as Coupled Oscillators

### 4.1 The Field Abstraction

HeteroFormer decomposes the input into four semantic fields:

| Field | Source | Semantic Role |
|-------|--------|--------------|
| **User** | User profile features (age, gender, location) | Static identity |
| **Item** | Item features (category, price, brand) | Target object |
| **Context** | Contextual features (time, device, page) | Situational modifier |
| **Sequence** | Historical behavior (from SSM + Prototype) | Temporal dynamics |

Rather than concatenating these fields into a single vector, HeteroFormer treats them as **interacting dynamical systems** on a shared feature lattice.

### 4.2 CrossFieldLayer: Attention + FiLM

Each CrossFieldLayer performs two operations:

**Self-Attention across Fields**:

```
Given field tensors F = [f_user, f_item, f_context, f_seq] ∈ R^(B×4×D)

Attn(Q=F·W_Q, K=F·W_K, V=F·W_V) = softmax(Q·K^T/√D) · V
```

This is standard multi-head self-attention, but applied **across fields rather than across sequence positions**. The attention weights A_{ij} represent "how much field i should attend to field j."

**FiLM Conditioning by Sequence**:

```
gamma = sigmoid(MLP(f_seq))   # [B, D]
beta  = MLP(f_seq)             # [B, D]

f_i' = f_i * gamma + beta     for i ∈ {user, item, context}
```

The sequence field acts as a **controller** that modulates the gain (gamma) and bias (beta) of all other fields. This is not feature interaction in the traditional sense (combining features to create new ones) but **feature modulation** (adjusting how features are expressed based on temporal context).

**Physical analogy**: The sequence field is the "magnetic field," and the other fields are "magnetizable materials." The material properties (user demographics, item attributes) are fixed, but their response to external stimuli is conditioned by the magnetic state (recent behavior history).

### 4.3 LangevinForceField: Stochastic Field Evolution

To model uncertainty in field interactions, HeteroFormer introduces a stochastic dynamics layer inspired by the Langevin equation from statistical mechanics:

```
dq_i/dt = p_i/m_i
dp_i/dt = -γ·p_i + F_i(q) + √(2γT)·ξ_i
```

where q_i is the position (field representation), p_i is the momentum, m_i is the mass, γ is the damping, T is the temperature, and ξ_i is white noise.

Discretized with symplectic Euler:

```
q_half = q + 0.5·dt·p/m
F_det = ForceNet(concat[q_half])    # Deterministic force from all fields
F = F_det + noise_scale·randn       # Add thermal noise
p = p·(1-γ·dt) + F·dt
q = q_half + 0.5·dt·p/m
```

**Learned parameters**:
- `mass_log`: per-field-per-dimension inertia. High mass means the field resists external forces (user identity is more stable than context).
- `gamma_log`: global damping. High damping leads to rapid equilibrium (short-term predictions); low damping allows oscillation (exploratory behavior).
- `temperature_log`: global temperature. High temperature increases stochasticity (diverse recommendations); low temperature sharpens focus (exploitation).

**Uncertainty head**: After Langevin evolution, an auxiliary network predicts per-field-per-dimension uncertainty:

```
σ_i = softplus(UncertaintyNet(concat[q_final]))
```

This uncertainty is used downstream to weight the contribution of each field to the final prediction.

---

## 5. Representation Surgery: Consistency Across Train/Eval Regimes

### 5.1 The Train-Eval Distribution Mismatch

A pervasive problem in multi-task architectures with generative components is **regime inconsistency**: during training, auxiliary tasks (diffusion, energy) inject stochasticity into the main task's input distribution; during evaluation, these tasks are disabled, causing a distribution shift.

In HeteroFormer's original design:

```python
if training:
    fused = gate0*proto + gate1*gen_diff + gate2*gen_energy   # stochastic
else:
    fused = proto                                              # deterministic
```

The CTR head learns a mapping from `fused` to click probability during training, but must generalize to `proto` during evaluation. This is not a domain adaptation problem (where source and target are related but different) but a **regime collapse problem** (where the evaluation input is a subset of the training input distribution).

### 5.2 Residual Fusion: Surgery Without Amputation

HeteroFormer resolves this through **residual representation surgery**:

```python
delta_diff = gen_diff - proto
delta_energy = gen_energy - proto
fused = proto + gate1*delta_diff + gate2*delta_energy
```

**Mathematical properties**:

| Property | Old (Weighted Average) | New (Residual) |
|----------|----------------------|----------------|
| Gate1=0, Gate2=0 | `fused = proto` via gate0=1 | `fused = proto` (identity) |
| Gate1>0 | diff "competes" with proto | diff "corrects" proto |
| Gradient flow to proto | Through all gates (competitive) | Direct + through residuals (cooperative) |
| Eval behavior | Hard switch: proto only | Soft continuation: proto + deterministic corrections |

The key insight is that the **residual terms delta_diff and delta_energy are learnable corrections**, not alternative representations. When the model is uncertain (high diffusion residual, flat energy landscape), the gates learn to suppress these corrections, gracefully defaulting to the prototype. When the model is confident (low residual, sharp energy boundaries), the gates amplify the corrections, enriching the representation.

### 5.3 Deterministic Evaluation Path

During evaluation, the stochastic components are replaced by their deterministic expectations:

| Component | Training | Evaluation |
|-----------|----------|------------|
| Diffusion timestep t | Uniform(0, T-1) | Fixed at T/2 (median noise level) |
| Diffusion noise ε | N(0, I) | 0 (mean) |
| Langevin noise ξ | N(0, I) | 0 (mean) |
| Sinkhorn perturbation | Gaussian(0, σ²) | 0 (mean) |

This ensures that `gen_diff` and `gen_energy` are **deterministic functions of proto**, not random variables. The CTR head sees the same geometric structure during train and eval, differing only in the "temperature" of the auxiliary tasks.

---

## 6. Online Uncertainty-Driven Scheduling: The MetaAligner

### 6.1 Beyond Validation-Dependent Scheduling

Traditional multi-task weight schedulers (DWA, GradNorm, PCGrad) require validation metrics to balance task losses. This creates a **latency problem**: in regimes with infrequent validation (e.g., per-epoch), the scheduler cannot respond to rapid training dynamics.

HeteroFormer's MetaAligner introduces a **dual-mode scheduler** that operates with or without validation signals:

### 6.2 Train-Schedule Mode: Uncertainty as Health Indicator

When validation is unavailable, MetaAligner extracts five uncertainty signals from the training forward pass (all computed in no_grad mode):

| Signal | Source | Interpretation |
|--------|--------|---------------|
| **Plateau** | train_auc variance over window | Optimization stagnation |
| **Grad Fatigue** | ctr_grad_norm decay ratio | Vanishing gradient dynamics |
| **Confidence Deficit** | logits_std below threshold | Over-conservative predictions |
| **Proto Chaos** | assignment_entropy / max_entropy | Prototype under-utilization |
| **Diff Residual** | MSE(pred_noise, noise) | Representation space roughness |

These signals are combined into a **training health score**:

```
health = Σ_i w_i · normalize(signal_i)
aux_weight = health · max_aux_weight   (clipped to [0, 0.5])
```

The health score directly maps to the total auxiliary task weight, with the diff/energy split determined by their relative learning efficiency (measured via probe loss trends).

### 6.3 Valid-PID Mode: Feedback Control

When validation is available, MetaAligner switches to a PID controller:

```
gap = train_auc - valid_auc          (generalization gap)
P = Kp · gap                          (proportional response)
I = Ki · ∫ gap dt                     (integral, anti-windup limited)
D = Kd · (-d(gap)/dt)                 (derivative, damping)
aux_weight = clip(P + I + D, 0, 0.5)
```

The integral term's anti-windup limiter (±5) prevents the controller from accumulating error during transient phases, ensuring recoverability.

### 6.4 Emergent Scheduling Behaviors

| Training Phase | Dominant Signal | MetaAligner Response | System Behavior |
|---------------|-----------------|---------------------|-----------------|
| Early convergence | Grad Fatigue ↑ | aux_weight ↑ | Inject diff noise to escape local minima |
| Mid plateau | Plateau ↑, Confidence ↓ | aux_weight ↑, energy ↑ | Sharpen decision boundaries via energy margin |
| Late overfitting | Gap ↑ | aux_weight ↑ (PID) | Strengthen regularization via all aux tasks |
| Healthy convergence | All signals ↓ | aux_weight ↓ | Focus on CTR, suppress auxiliary overhead |

---

## 7. The Unified Architecture: Sequence × Feature as Coupled Manifold Dynamics

### 7.1 Information Flow Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HeteroFormer Information Flow                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Temporal Input                                                             │
│      ↓                                                                      │
│  ┌────────────────────┐                                                     │
│  │ SSMCell            │  ← Continuous-time state evolution                  │
│  │ (Multi-scale decay)│     Δt as physical quantity, not index              │
│  └────────┬───────────┘                                                     │
│           ↓                                                                 │
│  ┌────────────────────┐                                                     │
│  │ Multi-scale Pooling│  ← short / long / static (physics-informed)        │
│  └────────┬───────────┘                                                     │
│           ↓                                                                 │
│  ┌────────────────────┐                                                     │
│  │ Prototype Manifold │  ← Riemannian geometry on unit sphere               │
│  │ (vMF + Sinkhorn)   │     User-conditioned rotation (Cayley)              │
│  └────────┬───────────┘                                                     │
│           ↓                                                                 │
│      proto_repr  ←──→  user_feat, item_feat, context_feat                  │
│           ↓              ↓                                                  │
│  ┌─────────────────────────────────────┐                                    │
│  │ CrossFieldNet                        │  ← Field dynamics as coupled system│
│  │ (Self-attn + SeqFiLM + Langevin)     │     Sequence modulates, not dominates│
│  └────────┬────────────────────────────┘                                    │
│           ↓                                                                 │
│  ┌─────────────────────────────────────┐                                    │
│  │ Representation Surgery               │  ← Residual fusion with consistency │
│  │ (proto + gate·delta_diff + gate·delta_energy)                        │
│  └────────┬────────────────────────────┘                                    │
│           ↓                                                                 │
│       fused_repr → CTRHead → Prediction                                     │
│                                                                             │
│  MetaAligner (peripheral nervous system)                                    │
│      ↓                                                                      │
│  Monitors: plateau, fatigue, confidence, chaos, residual                    │
│  Acts:     modulates aux_weight → gates → fusion balance                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Coupling Points: Where Sequence and Feature Truly Meet

**Coupling Point I: Time → Geometry**

The SSM's output `s_short` is not merely a sequence summary; it is the **initial condition** for the prototype manifold. The concentration parameter κ is directly computed from the sequence's temporal statistics (recency, frequency), meaning that the **geometric sharpness of prototype assignment is determined by temporal dynamics**.

**Coupling Point II: Geometry → Field Dynamics**

The prototype representation `proto_repr` enters CrossFieldNet as the **sequence field**. But it does not interact symmetrically with other fields; instead, it acts as the **conditioning signal** for FiLM modulation. Thus, the geometric structure of the manifold (which prototypes are active, how sharply they are assigned) directly controls the **gain and bias** of all other feature fields.

**Coupling Point III: Field Dynamics → Uncertainty**

The LangevinForceField's output includes per-field uncertainty estimates. These uncertainties feed back into the MetaAligner's health score: if the sequence field has high uncertainty (noisy Langevin evolution), MetaAligner reduces the auxiliary weight to prevent unstable corrections from contaminating the main task.

**Coupling Point IV: Uncertainty → Surgery**

The MetaAligner's output `aux_weight` controls whether `gen_diff` and `gen_energy` are computed and backpropagated. But the residual fusion ensures that even when `aux_weight = 0`, the eval path remains consistent. The surgery is **minimally invasive**: it corrects the representation only when the patient (the model) needs it.

---

## 8. Discussion: Architectural Philosophy

### 8.1 Against Modularity

HeteroFormer deliberately violates the principle of strict modularity. In conventional design, one might compose:
- A sequence module (SASRec)
- A feature interaction module (DeepFM)
- A multi-task module (MMoE)

Each module is independently developed, tested, and replaced. HeteroFormer argues that **for the specific problem of sequential feature interaction, modularity is a liability**. The sequence encoder must know that its output will be used for manifold assignment; the manifold must know that its prototypes will be rotated by user features; the feature interaction must know that the sequence field acts as a conditioner, not a peer.

This is not monolithic design but **organic design**: each component is shaped by its relational role in the whole.

### 8.2 For Differentiability

Every design choice in HeteroFormer is differentiable end-to-end:
- Time warping: differentiable through softplus and log.
- Cayley rotation: differentiable through matrix inversion (with regularization).
- Sinkhorn iteration: differentiable through implicit differentiation (or unrolled iterations).
- Langevin dynamics: differentiable through the reparameterization trick (noise as external input).
- MetaAligner: differentiable through the dummy parameter (though in practice, the scheduler operates via Python control flow, not autograd).

This ensures that gradients flow from the final prediction all the way back to the raw time values and feature embeddings, enabling **co-adaptation** rather than **staged optimization**.

### 8.3 Toward Physical Interpretability

HeteroFormer's components are named after physical concepts (state space, force field, Langevin, energy landscape) not for metaphorical flair, but because the mathematics genuinely correspond:
- The SSM is a physical system with inertia and damping.
- The prototypes are energy minima on a potential surface.
- The Langevin dynamics add thermal noise to escape local minima.
- The MetaAligner is a thermostat that regulates system temperature.

This physical grounding provides **intuitive debugging**: if the model overfits, one checks the "temperature" (aux_weight); if it underfits, one checks the "mass" (field inertia); if prototypes collapse, one checks the "entropy" (Sinkhorn regularization).

---

## 9. Conclusion

HeteroFormer presents a unified framework for sequential feature interaction in conversion rate prediction, built on three foundational shifts:

1. **From discrete to continuous time**: State space models with learned time warping treat temporal intervals as physical quantities, enabling natural handling of irregular sampling and multi-scale dynamics.

2. **From Euclidean embedding to Riemannian manifolds**: Prototype assignment via user-conditioned Cayley rotations and entropy-regularized optimal transport embeds feature interactions on structured geometric spaces with explicit uncertainty modeling.

3. **From competitive to cooperative fusion**: Residual representation surgery ensures train/eval consistency by treating auxiliary representations as corrections to a stable baseline, rather than alternatives that compete for dominance.

These shifts are orchestrated by a dual-mode task scheduler that operates via online uncertainty signals when validation is unavailable and via classical feedback control when it is available. The result is an architecture where sequence modeling and feature interaction are not concatenated stages but **coupled dynamical systems** evolving on a shared geometric substrate.

---

*This technical report describes the architectural principles and mathematical foundations of the HeteroFormer model family. Implementation details, hyperparameter configurations, and empirical evaluations are documented separately.*

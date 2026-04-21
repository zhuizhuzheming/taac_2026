"""
TokenFormer: Unify the Multi-Field and Sequential Recommendation Worlds
PyTorch Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for unified token stream.
    论文Sec 4.1.1和Appendix D.1描述：使用乘法RoPE避免加性位置编码的秩塌陷问题
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算旋转角度
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # [max_seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """将输入的后半部分旋转"""
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x: torch.Tensor, seq_len: int, pos_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_heads, seq_len, head_dim]
            seq_len: 序列长度
            pos_ids: 位置索引 [batch_size, seq_len]，如果为None则使用默认位置
        """
        if pos_ids is None:
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        else:
            # 根据pos_ids获取对应位置编码
            cos = self.cos_cached[:, :, pos_ids, :]
            sin = self.sin_cached[:, :, pos_ids, :]
        
        return x * cos + self.rotate_half(x) * sin


class BFTSAttention(nn.Module):
    """
    Bottom-Full-Top-Sliding (BFTS) Attention Mechanism
    论文Sec 4.3描述：
    - 浅层使用Full Causal Attention
    - 深层使用Shrinking Sliding Window Attention
    - 深层丢弃非序列token的注意力
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_idx: int,
        num_full_layers: int,
        num_swa_layers: int,
        window_sizes: List[int],
        dropout: float = 0.1,
        use_rope: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layer_idx = layer_idx
        self.num_full_layers = num_full_layers
        self.num_swa_layers = num_swa_layers
        self.window_sizes = window_sizes
        
        assert len(window_sizes) == num_swa_layers, "window_sizes长度必须等于num_swa_layers"
        
        # 判断当前层类型
        self.is_full_layer = layer_idx < num_full_layers
        if not self.is_full_layer:
            swa_idx = layer_idx - num_full_layers
            self.window_size = window_sizes[swa_idx]
        else:
            self.window_size = float('inf')
        
        # Q, K, V投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim) if use_rope else None
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def create_swa_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """创建Sliding Window Attention mask"""
        # 基础因果mask
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        # 每个token只能看到最近的window_size个token
        for i in range(seq_len):
            start = max(0, i - window_size)
            mask[i, start:i+1] = 0.0
        
        # 上三角保持-inf（因果性）
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    def create_discard_mask(
        self, 
        seq_len: int, 
        num_static_tokens: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        非序列token丢弃策略 (论文Sec 4.3)
        深层不再关注前M个非序列token
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        # 深层：所有token都不能看到静态token
        mask[:, :num_static_tokens] = float('-inf')
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        num_static_tokens: int = 0,
        pos_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, dim]
            num_static_tokens: 非序列token数量M
            pos_ids: 位置索引 [batch_size, seq_len]
            attention_mask: 额外的attention mask
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Q, K, V投影
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # 多头拆分 [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用RoPE
        if self.rope is not None:
            Q = self.rope(Q, seq_len, pos_ids)
            K = self.rope(K, seq_len, pos_ids)
        
        # 计算attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]
        
        # 创建visibility mask
        if self.is_full_layer:
            # Full causal attention
            visibility_mask = self.create_causal_mask(seq_len, hidden_states.device)
        else:
            # Sliding window attention
            visibility_mask = self.create_swa_mask(seq_len, self.window_size, hidden_states.device)
            
            # 非序列token丢弃 (仅深层)
            if num_static_tokens > 0:
                discard_mask = self.create_discard_mask(seq_len, num_static_tokens, hidden_states.device)
                visibility_mask = visibility_mask + discard_mask
        
        # 应用visibility mask
        visibility_mask = visibility_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        scores = scores + visibility_mask
        
        # 额外的attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax和dropout
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_weights = self.dropout(attn_weights)
        
        # 加权聚合
        attn_output = torch.matmul(attn_weights, V)  # [batch, heads, seq, head_dim]
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class NonLinearInteractionRepresentation(nn.Module):
    """
    Non-Linear Interaction Representation (NLIR)
    论文Sec 4.4描述：
    - 从输入计算门控G
    - 使用sigmoid激活
    - 与attention输出做逐元素乘法
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
    
    def forward(self, attn_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_output: Attention输出 A
            hidden_states: 层输入 X (用于计算门控)
        Returns:
            非线性交互表示: σ(G) ⊙ A
        """
        # 计算门控 G = X * W_g
        gate = self.gate_proj(hidden_states)
        # Sigmoid激活
        gate = torch.sigmoid(gate)
        # 逐元素乘法
        interacted = gate * attn_output
        return interacted


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network
    论文Sec 4.5描述
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(xW1) ⊙ (xW2) * W3
        swish = F.silu(self.w1(x))
        gated = swish * self.w2(x)
        output = self.w3(gated)
        output = self.dropout(output)
        return output


class UnifiedInteractionBlock(nn.Module):
    """
    Unified Interaction Block (UIB)
    论文Sec 4.2和Figure 2描述的核心模块
    包含BFTS Attention + NLIR + SwiGLU FFN
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_idx: int,
        num_full_layers: int,
        num_swa_layers: int,
        window_sizes: List[int],
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_rms_norm: bool = True
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        # BFTS Attention
        self.attn = BFTSAttention(
            dim=dim,
            num_heads=num_heads,
            layer_idx=layer_idx,
            num_full_layers=num_full_layers,
            num_swa_layers=num_swa_layers,
            window_sizes=window_sizes,
            dropout=dropout
        )
        
        # NLIR
        self.nlir = NonLinearInteractionRepresentation(dim)
        
        # SwiGLU FFN
        self.ffn = SwiGLU(dim, ffn_hidden_dim, dropout)
        
        # Normalization
        if use_rms_norm:
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        num_static_tokens: int = 0,
        pos_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, dim]
            num_static_tokens: 非序列token数量
            pos_ids: 位置索引
            attention_mask: 额外mask
        """
        # 第一层Norm
        normed = self.norm1(hidden_states)
        
        # BFTS Attention
        attn_output = self.attn(
            normed,
            num_static_tokens=num_static_tokens,
            pos_ids=pos_ids,
            attention_mask=attention_mask
        )
        
        # NLIR: σ(G) ⊙ A
        interacted = self.nlir(attn_output, normed)
        
        # 残差连接: I = X + σ(G) ⊙ A
        hidden_states = hidden_states + interacted
        
        # FFN + 残差
        normed2 = self.norm2(hidden_states)
        ffn_output = self.ffn(normed2)
        hidden_states = hidden_states + ffn_output  # X^(l+1) = I + H
        
        return hidden_states


class RMSNorm(nn.Module):
    """RMSNorm归一化"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class TokenFormer(nn.Module):
    """
    TokenFormer: 统一多字段和序列推荐的完整模型
    论文Sec 4描述的整体架构
    """
    def __init__(
        self,
        num_fields: int,  # M: 非序列字段数量
        seq_len: int,  # T: 历史序列长度
        num_targets: int,  # K: 目标item数量
        vocab_size: int,  # 嵌入词表大小
        dim: int = 256,  # 隐藏维度
        num_layers: int = 4,  # 总层数L
        num_heads: int = 4,  # 注意力头数
        num_full_layers: int = 2,  # 全注意力层数lf
        window_sizes: List[int] = [32, 16],  # 滑动窗口大小(从大到小)
        ffn_mult: int = 4,  # FFN隐藏层倍数
        dropout: float = 0.1,
        action_space_size: int = 2,  # 动作空间大小(CTR预测通常为2:点击/未点击)
        use_actions: bool = True,  # 是否使用action-aware模式
        use_sep_token: bool = True,  # 是否使用分隔token
    ):
        super().__init__()
        self.num_fields = num_fields
        self.seq_len = seq_len
        self.num_targets = num_targets
        self.dim = dim
        self.num_layers = num_layers
        self.num_full_layers = num_full_layers
        self.num_swa_layers = num_layers - num_full_layers
        self.use_actions = use_actions
        self.use_sep_token = use_sep_token
        
        # 验证window_sizes
        assert len(window_sizes) == self.num_swa_layers, \
            f"window_sizes长度({len(window_sizes)})必须等于SWA层数({self.num_swa_layers})"
        
        # 计算总序列长度
        # S_L = M + T + K + N_sep (without actions)
        # S_L = M + 2T + K + N_sep (with actions)
        self.num_sep = 2 if use_sep_token else 0
        if use_actions:
            self.total_seq_len = num_fields + 2 * seq_len + num_targets + self.num_sep
        else:
            self.total_seq_len = num_fields + seq_len + num_targets + self.num_sep
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # 分隔token嵌入
        if use_sep_token:
            self.sep_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 类型感知位置分配 (论文Sec 4.1.1)
        # 静态字段: pos=0
        # 行为token: pos=实际时间位置
        # 目标token: pos=S_L+1
        self.register_buffer('pos_assignments', self._create_pos_assignments())
        
        # 堆叠的Unified Interaction Blocks
        self.layers = nn.ModuleList([
            UnifiedInteractionBlock(
                dim=dim,
                num_heads=num_heads,
                layer_idx=i,
                num_full_layers=num_full_layers,
                num_swa_layers=self.num_swa_layers,
                window_sizes=window_sizes,
                ffn_hidden_dim=dim * ffn_mult,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # 最终Norm
        self.final_norm = RMSNorm(dim)
        
        # 预测头
        self.prediction_head = nn.Linear(dim, action_space_size)
        
        # 初始化
        self._init_weights()
    
    def _create_pos_assignments(self) -> torch.Tensor:
        """创建类型感知位置分配"""
        pos = torch.zeros(self.total_seq_len, dtype=torch.long)
        idx = 0
        
        # 静态字段: pos=0
        for _ in range(self.num_fields):
            pos[idx] = 0
            idx += 1
        
        # 分隔token
        if self.use_sep_token:
            pos[idx] = 0
            idx += 1
        
        # 序列token: pos=时间位置
        if self.use_actions:
            for t in range(self.seq_len):
                pos[idx] = t + 1  # 状态token
                idx += 1
                pos[idx] = t + 1  # action token
                idx += 1
        else:
            for t in range(self.seq_len):
                pos[idx] = t + 1
                idx += 1
        
        # 分隔token
        if self.use_sep_token:
            pos[idx] = self.total_seq_len + 1
            idx += 1
        
        # 目标token: pos=S_L+1
        for _ in range(self.num_targets):
            pos[idx] = self.total_seq_len + 1
            idx += 1
        
        return pos
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.normal_(self.embedding.weight, std=0.02)
        if self.use_sep_token:
            nn.init.normal_(self.sep_token, std=0.02)
    
    def create_input_stream(
        self,
        field_features: torch.Tensor,  # [batch, M]
        seq_states: torch.Tensor,  # [batch, T] 或 [batch, T, 2] (with actions)
        seq_actions: Optional[torch.Tensor] = None,  # [batch, T] (with actions)
        target_features: torch.Tensor = None,  # [batch, K]
    ) -> torch.Tensor:
        """
        构建统一token流 (论文Eq. 7)
        
        X^(0) = [x_1^F, ..., x_M^F, <sep>, x_s1^T, x_a1^T, ..., x_sT^T, x_aT^T, <sep>, x_c1^V, ..., x_cK^V]
        """
        batch_size = field_features.size(0)
        embeddings = []
        
        # 非序列字段嵌入
        field_emb = self.embedding(field_features)  # [batch, M, dim]
        embeddings.append(field_emb)
        
        # 分隔token
        if self.use_sep_token:
            sep = self.sep_token.expand(batch_size, 1, self.dim)
            embeddings.append(sep)
        
        # 序列token嵌入
        if self.use_actions and seq_actions is not None:
            # action-aware: 交替状态和行为
            for t in range(self.seq_len):
                state_emb = self.embedding(seq_states[:, t:t+1])  # [batch, 1, dim]
                embeddings.append(state_emb)
                action_emb = self.embedding(seq_actions[:, t:t+1])  # [batch, 1, dim]
                embeddings.append(action_emb)
        else:
            seq_emb = self.embedding(seq_states)  # [batch, T, dim]
            embeddings.append(seq_emb)
        
        # 分隔token
        if self.use_sep_token:
            embeddings.append(sep)
        
        # 目标token嵌入
        if target_features is not None:
            target_emb = self.embedding(target_features)  # [batch, K, dim]
            embeddings.append(target_emb)
        
        # 拼接
        x = torch.cat(embeddings, dim=1)  # [batch, S_L, dim]
        return x
    
    def forward(
        self,
        field_features: torch.Tensor,
        seq_states: torch.Tensor,
        seq_actions: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_indices: Optional[List[int]] = None,  # 用于计算loss的token索引
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            field_features: [batch, M] 非序列特征ID
            seq_states: [batch, T] 序列状态ID
            seq_actions: [batch, T] 序列行为ID (action-aware模式)
            target_features: [batch, K] 目标特征ID
            attention_mask: 额外mask
            loss_indices: 需要计算loss的token位置索引
        """
        batch_size = field_features.size(0)
        
        # 构建统一token流
        hidden_states = self.create_input_stream(
            field_features, seq_states, seq_actions, target_features
        )
        
        # 扩展位置ID到batch维度
        pos_ids = self.pos_assignments.unsqueeze(0).expand(batch_size, -1)
        
        # 通过堆叠的UIB层
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                num_static_tokens=self.num_fields + (1 if self.use_sep_token else 0),
                pos_ids=pos_ids,
                attention_mask=attention_mask
            )
        
        # 最终归一化
        hidden_states = self.final_norm(hidden_states)
        
        # 预测
        logits = self.prediction_head(hidden_states)  # [batch, S_L, action_space]
        
        outputs = {
            'hidden_states': hidden_states,
            'logits': logits,
        }
        
        # 如果指定了loss_indices，提取对应的logits
        if loss_indices is not None:
            loss_logits = logits[:, loss_indices, :]  # [batch, num_loss, action_space]
            outputs['loss_logits'] = loss_logits
        
        return outputs
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        计算Cross-Entropy Loss (论文Sec 4.6)
        
        Args:
            logits: [batch, seq_len, num_actions] 或 [batch, num_loss, num_actions]
            labels: [batch, seq_len] 或 [batch, num_loss] 真实标签
            loss_indices: 如果logits已经是提取过的
        """
        if loss_indices is not None:
            # 从完整logits中提取
            loss_logits = logits[:, loss_indices, :]
        else:
            loss_logits = logits
        
        # CE Loss
        loss = F.cross_entropy(
            loss_logits.view(-1, loss_logits.size(-1)),
            labels.view(-1),
            reduction='mean'
        )
        return loss

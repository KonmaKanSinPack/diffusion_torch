"""
DiT (Diffusion Transformer) — 用于图像扩散模型的 Transformer 骨干网络
=======================================================================
参考论文:
  - Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
  - Ma et al., "SiT: Exploring Flow and Diffusion-based Generative Models
    with Scalable Interpolant Transformers", ECCV 2024

核心设计:
  1. Patch Embedding: 将图像分割为不重叠的 patch 并映射到 token 维度
  2. adaLN-Zero 条件注入: 时间步嵌入通过 Adaptive LayerNorm 调制每层参数
  3. Zero-Init 输出层: 确保初始化时每个 DiT 块约等于恒等映射, 训练更稳定
  4. QK-Norm: 在 Attention 中对 query/key 做 LayerNorm, 防止注意力权重发散
  5. 2D 正弦位置编码: 为 patch 提供绝对位置信息

支持三种模型规模:
  - DiT-T (Tiny):  D=192,  depth=6,  heads=3,  ~5M params
  - DiT-S (Small): D=384,  depth=12, heads=6,  ~33M params
  - DiT-B (Base):  D=768,  depth=12, heads=12, ~130M params
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN 调制: y = (1 + scale) * LN(x) + shift
    x: [B, N, D],  shift/scale: [B, D]
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# 时间步嵌入
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """正弦位置编码 → 两层 MLP, 将离散时间步映射为连续向量"""

    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        """标准正弦 / 余弦位置编码, 与 Vaswani et al. (2017) 一致"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.freq_embed_size)
        return self.mlp(t_freq)


# ---------------------------------------------------------------------------
# Patch 嵌入
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """将图像分割为不重叠的 patch 并通过卷积投影到 token 维度
    Image [B, C, H, W] → Tokens [B, N, D],  其中 N = (H/P) × (W/P)
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        # 使用不重叠的卷积作为 patch 投影 (等价于 Linear(flatten(patch)))
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)                       # [B, D, H_p, W_p]
        x = x.flatten(2).transpose(1, 2)       # [B, N, D]
        return x


# ---------------------------------------------------------------------------
# 多头自注意力 (带 QK-Norm)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """多头自注意力, 在 query 和 key 上附加 LayerNorm (QK-Norm) 以稳定训练"""

    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} 不能被 num_heads={num_heads} 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

        # QK-Norm: 防止高维 attention logit 爆炸
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)                     # 各 [B, heads, N, head_dim]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot-product attention
        # 如果 PyTorch >= 2.0, 可以使用 F.scaled_dot_product_attention 加速
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# MLP (前馈网络)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """标准 Transformer 前馈网络: Linear → GELU → Linear"""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# DiT 块: adaLN-Zero 条件化的 Transformer 块
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    DiT Transformer 块, 使用 adaLN-Zero 条件注入.

    与标准 Transformer 块的区别:
      - LayerNorm 的 affine 参数由条件向量 (时间步嵌入) 动态生成
      - 注意力和 MLP 输出乘以 gate 系数 (初始化为 0)
      - 初始化时, 整个块近似恒等映射, 有利于深层网络训练
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = MLP(hidden_size, mlp_hidden)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # adaLN-Zero: 从条件向量生成 6 个调制参数
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # 关键: Zero-Init — 使 gate 初始为 0, 块输出为恒等
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]  — token 序列
        c: [B, D]     — 条件向量 (时间步嵌入)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # 自注意力分支
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h = self.attn(h)
        h = self.dropout(h)
        x = x + gate_msa.unsqueeze(1) * h

        # 前馈网络分支
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        h = self.dropout(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


# ---------------------------------------------------------------------------
# 输出层: adaLN → Linear → Unpatchify
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """DiT 输出层: adaLN 调制 → 线性投影到 patch 像素空间"""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        # Zero-Init: 输出层初始化为零, 初始预测为全零
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# 2D 正弦位置编码工具
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    """1D 正弦余弦位置编码"""
    assert embed_dim % 2 == 0
    pos = np.arange(length, dtype=np.float64)
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    out = np.outer(pos, omega)  # [length, embed_dim//2]
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # [length, embed_dim]


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """2D 正弦余弦位置编码 (分别对 H 和 W 维度编码后拼接)"""
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h))  # [2, grid_size, grid_size]
    emb_h = _get_1d_sincos_pos_embed(half, grid_size)  # [grid_size, half]
    emb_w = _get_1d_sincos_pos_embed(half, grid_size)  # [grid_size, half]
    # 为每个 (h, w) 位置拼接两个 1D 编码
    emb_h = np.repeat(emb_h, grid_size, axis=0)         # [N, half], 行重复
    emb_w = np.tile(emb_w, (grid_size, 1))              # [N, half], 列重复
    return np.concatenate([emb_h, emb_w], axis=1)        # [N, embed_dim]


# ---------------------------------------------------------------------------
# DiT 主模型
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT)

    输入: 带噪图像 x_t ∈ ℝ^{B×C×H×W} 和时间步 t ∈ ℤ^{B}
    输出: 预测结果 (噪声 ε 或 速度 v 或 x0) ∈ ℝ^{B×C_out×H×W}

    架构流程:
      1. PatchEmbed: Image → Tokens [B, N, D]   (N = (H/P)²)
      2. 加上 2D 正弦位置编码
      3. N 个 DiT Block (adaLN-Zero 条件化)
      4. FinalLayer → Unpatchify → 输出图像
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        # ---- Patch 嵌入 ----
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embed.num_patches

        # ---- 可学习的位置嵌入 (用正弦编码初始化) ----
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # ---- 时间步嵌入 ----
        self.t_embedder = TimestepEmbedder(hidden_size)

        # ---- Transformer 主体 ----
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        # ---- 输出层 ----
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

        # ---- 权重初始化 ----
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """精心设计的初始化策略, 确保训练初期稳定性"""
        # 位置编码: 2D 正弦初始化
        pos_embed = _get_2d_sincos_pos_embed(
            self.hidden_size, self.patch_embed.grid_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Patch 投影: Xavier 均匀初始化
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

        # 时间步 MLP: 标准初始化
        for m in self.t_embedder.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Transformer 块内部
        for block in self.blocks:
            # QKV 和输出投影
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            if block.attn.qkv.bias is not None:
                nn.init.zeros_(block.attn.qkv.bias)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            nn.init.zeros_(block.attn.proj.bias)
            # MLP
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.zeros_(block.mlp.fc1.bias)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            nn.init.zeros_(block.mlp.fc2.bias)
            # adaLN 已在 DiTBlock.__init__ 中 zero-init

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """将 token 序列还原为图像
        x: [B, N, P*P*C_out] → [B, C_out, H, W]
        """
        C = self.out_channels
        P = self.patch_size
        h = w = self.patch_embed.grid_size
        # [B, h, w, P, P, C]
        x = x.reshape(-1, h, w, P, P, C)
        # 重排: [B, C, h, P, w, P]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        # 合并空间维度: [B, C, H, W]
        x = x.reshape(-1, C, h * P, w * P)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
          x: [B, C_in, H, W] — 带噪图像
          t: [B]              — 整数时间步 (0 ~ T-1)

        返回:
          pred: [B, C_out, H, W] — 模型预测 (噪声 / 速度 / x0)
        """
        B = x.shape[0]

        # 1. Patch 嵌入 + 位置编码
        x = self.patch_embed(x) + self.pos_embed          # [B, N, D]

        # 2. 时间步条件向量
        c = self.t_embedder(t)                              # [B, D]

        # 3. Transformer 块处理
        for block in self.blocks:
            x = block(x, c)

        # 4. 输出层 → 还原为图像
        x = self.final_layer(x, c)                          # [B, N, P*P*C_out]
        x = self.unpatchify(x)                              # [B, C_out, H, W]

        return x


# ---------------------------------------------------------------------------
# 预定义配置
# ---------------------------------------------------------------------------

def DiT_T(img_size: int = 32, in_channels: int = 3, out_channels: int = 3, **kw) -> DiT:
    """DiT-Tiny: ~5M 参数, 适合快速实验"""
    return DiT(
        img_size=img_size, patch_size=2,
        in_channels=in_channels, out_channels=out_channels,
        hidden_size=192, depth=6, num_heads=3, **kw,
    )


def DiT_S(img_size: int = 32, in_channels: int = 3, out_channels: int = 3, **kw) -> DiT:
    """DiT-Small: ~33M 参数, CIFAR-10 主力配置"""
    return DiT(
        img_size=img_size, patch_size=2,
        in_channels=in_channels, out_channels=out_channels,
        hidden_size=384, depth=12, num_heads=6, **kw,
    )


def DiT_B(img_size: int = 32, in_channels: int = 3, out_channels: int = 3, **kw) -> DiT:
    """DiT-Base: ~130M 参数, 追求极致质量"""
    return DiT(
        img_size=img_size, patch_size=2,
        in_channels=in_channels, out_channels=out_channels,
        hidden_size=768, depth=12, num_heads=12, **kw,
    )

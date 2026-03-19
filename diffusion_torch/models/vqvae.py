"""
多尺度 VQVAE (Multi-Scale Vector Quantized VAE)

参考 VAR (Visual Autoregressive Modeling) 的多尺度量化设计：
- 图像被编码为多个分辨率层级的离散 token map (如 1×1, 2×2, 4×4, ..., 16×16)
- 粗尺度（低分辨率）自然捕获低频信息 = SVD 的角色（轮廓、外观）
- 细尺度（高分辨率）捕获高频信息 = Diffusion 的角色（纹理、光照）
- 每个尺度拥有独立的 codebook，形成层级化频率分解

架构设计:
  Encoder → 多尺度 Feature Maps → Vector Quantization (per scale) → Decoder
  
  多尺度量化的核心思想:
    对 encoder 输出在不同分辨率进行自适应池化，得到 {r_1, r_2, ..., r_K} 尺度的特征图，
    每个尺度独立做向量量化。重建时，从最粗尺度逐级上采样并累加，形成 coarse-to-fine 分解。
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  配置
# ============================================================

@dataclass
class VQVAEConfig:
    """Multi-Scale VQVAE 配置"""
    # 输入图像
    image_size: int = 32          # 图像尺寸
    in_channels: int = 3          # 输入通道数

    # Encoder/Decoder
    hidden_dim: int = 256         # Encoder/Decoder 的隐藏通道数
    latent_dim: int = 32          # 量化潜在维度（每个 code vector 的维度）
    num_res_blocks: int = 2       # 每个阶段的残差块数量
    ch_mult: List[int] = field(default_factory=lambda: [1, 2, 4])  # 通道倍增

    # 多尺度量化
    # VAR 风格: 从 1×1 到 latent_size×latent_size 的多个尺度
    # 对于 32×32 图像, encoder 下采样 4x → 8×8 latent
    # 尺度: [1, 2, 4, 8]
    multi_scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    codebook_size: int = 1024     # 每个尺度的 codebook 大小
    commitment_weight: float = 0.25  # commitment loss 权重
    codebook_ema_decay: float = 0.99  # EMA codebook 更新衰减率

    # 量化器类型: 'ema' 或 'vanilla'
    quantizer_type: str = 'ema'

    @property
    def latent_size(self) -> int:
        """Encoder 输出的空间尺寸"""
        downsample_factor = 2 ** (len(self.ch_mult) - 1)
        return self.image_size // downsample_factor

    @property
    def num_scales(self) -> int:
        return len(self.multi_scales)


# ============================================================
#  基础模块
# ============================================================

class ResBlock(nn.Module):
    """残差块，带可选的通道变换"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        # shortcut: 如果通道数不同，用 1×1 卷积匹配
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.net(x)


class Downsample(nn.Module):
    """2x 下采样（步长2的卷积，避免信息丢失）"""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x 上采样（最近邻插值 + 卷积平滑）"""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ============================================================
#  Encoder
# ============================================================

class Encoder(nn.Module):
    """
    多阶段卷积 Encoder
    输入: (B, C, H, W) → 输出: (B, latent_dim, H/4, W/4)
    
    结构: Conv_in → [ResBlock * N + Downsample] * (levels-1) → [ResBlock * N] → Conv_out
    """

    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.hidden_dim

        # 初始卷积
        self.conv_in = nn.Conv2d(cfg.in_channels, ch, 3, padding=1)

        # 下采样阶段
        self.down_blocks = nn.ModuleList()
        in_ch = ch
        for i, mult in enumerate(cfg.ch_mult):
            out_ch = ch * mult
            block = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                block.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            # 除了最后一个阶段，都做下采样
            if i < len(cfg.ch_mult) - 1:
                block.append(Downsample(out_ch))
            self.down_blocks.append(block)

        # 中间层
        self.mid = nn.Sequential(
            ResBlock(in_ch, in_ch),
            ResBlock(in_ch, in_ch),
        )

        # 输出投影到 latent_dim
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, cfg.latent_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            z: (B, latent_dim, H_lat, W_lat)  连续潜在表示
        """
        h = self.conv_in(x)
        for block_list in self.down_blocks:
            for block in block_list:
                h = block(h)
        h = self.mid(h)
        z = self.conv_out(h)
        return z


# ============================================================
#  Decoder
# ============================================================

class Decoder(nn.Module):
    """
    多阶段卷积 Decoder (Encoder 的镜像)
    输入: (B, latent_dim, H_lat, W_lat) → 输出: (B, C, H, W)
    """

    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.hidden_dim

        # ch_mult 逆序，用于上采样
        ch_mult_rev = list(reversed(cfg.ch_mult))
        in_ch = ch * ch_mult_rev[0]

        # 输入投影
        self.conv_in = nn.Conv2d(cfg.latent_dim, in_ch, 1)

        # 中间层
        self.mid = nn.Sequential(
            ResBlock(in_ch, in_ch),
            ResBlock(in_ch, in_ch),
        )

        # 上采样阶段
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(ch_mult_rev):
            out_ch = ch * mult
            block = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                block.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            # 除了最后一个阶段，都做上采样
            if i < len(ch_mult_rev) - 1:
                block.append(Upsample(out_ch))
                # 上采样后通道数设为下一阶段
                in_ch = out_ch
            self.up_blocks.append(block)

        # 输出卷积
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, cfg.in_channels, 3, padding=1),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: (B, latent_dim, H_lat, W_lat)  量化后的潜在表示
        Returns:
            x_recon: (B, C, H, W)  重建图像
        """
        h = self.conv_in(z_q)
        h = self.mid(h)
        for block_list in self.up_blocks:
            for block in block_list:
                h = block(h)
        x_recon = self.conv_out(h)
        return x_recon


# ============================================================
#  向量量化器
# ============================================================

class EMAVectorQuantizer(nn.Module):
    """
    EMA (Exponential Moving Average) 更新的向量量化器
    
    相比 vanilla VQ (直接梯度更新 codebook)，EMA 更新更稳定:
    - codebook 向量通过 EMA 跟踪被分配的 encoder 输出的均值
    - 不需要 codebook loss，只需 commitment loss (让 encoder 输出接近 codebook)
    
    Straight-Through 梯度估计:
    - 前向: z_q = codebook[argmin distance]  (离散、不可导)
    - 反向: 梯度直接从 z_q 复制到 z_e  (straight-through estimator)
    """

    def __init__(self, codebook_size: int, latent_dim: int, decay: float = 0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.decay = decay

        # codebook: (K, D) — 每个 code vector 维度为 latent_dim
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA 统计量 (不参与梯度计算)
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_embedding_sum', self.embedding.weight.clone())

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D, H, W) 连续潜在表示
        Returns:
            z_q:    (B, D, H, W) 量化后的潜在表示 (带 straight-through 梯度)
            indices: (B, H*W)    codebook 索引
            commit_loss: scalar  commitment loss
        """
        B, D, H, W = z.shape

        # (B, D, H, W) → (B*H*W, D)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        # 计算距离: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=False)
            - 2.0 * z_flat @ self.embedding.weight.t()
        )  # (B*H*W, K)

        # 最近邻查找
        indices = dist.argmin(dim=1)  # (B*H*W,)
        z_q_flat = self.embedding(indices)  # (B*H*W, D)

        # EMA 更新 codebook (仅训练时)
        if self.training:
            # one-hot 编码
            encodings = F.one_hot(indices, self.codebook_size).float()  # (B*H*W, K)

            # 更新簇大小
            self.ema_cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            # 更新嵌入求和
            self.ema_embedding_sum.mul_(self.decay).add_(
                encodings.t() @ z_flat, alpha=1 - self.decay
            )

            # Laplace 平滑防止空簇
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.codebook_size * 1e-5)
                * n
            )

            # 更新 codebook
            self.embedding.weight.data.copy_(
                self.ema_embedding_sum / cluster_size.unsqueeze(1)
            )

        # commitment loss: 让 encoder 输出接近 codebook 向量
        commit_loss = F.mse_loss(z_flat, z_q_flat.detach())

        # Straight-Through Estimator: 前向用量化值，反向梯度传给 z
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        # 恢复形状
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)
        indices = indices.reshape(B, H * W)

        return z_q, indices, commit_loss


# ============================================================
#  多尺度量化器 (VAR 的核心思想)
# ============================================================

class MultiScaleQuantizer(nn.Module):
    """
    多尺度向量量化器 — VAR 模型的核心设计
    
    对 encoder 输出的连续 latent map 在多个分辨率进行量化:
      scale 1: 1×1 → 全局语义 (最低频)
      scale 2: 2×2 → 粗略结构
      scale 3: 4×4 → 中等细节
      scale K: 8×8 → 精细纹理 (最高频)
    
    每个尺度:
    1. 将 latent map 自适应池化到目标分辨率
    2. 减去前面所有尺度的上采样累积 (残差量化)
    3. 独立做向量量化
    
    这样自然形成了 coarse-to-fine 的频率分解:
    - 粗尺度 codebook 捕获低频 (轮廓、颜色)
    - 细尺度 codebook 捕获高频残差 (纹理、边缘)
    """

    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.scales = cfg.multi_scales
        self.latent_dim = cfg.latent_dim

        # 每个尺度一个独立的 VQ
        self.quantizers = nn.ModuleList([
            EMAVectorQuantizer(cfg.codebook_size, cfg.latent_dim, cfg.codebook_ema_decay)
            for _ in self.scales
        ])

        # 每个尺度对应的 残差投影 (可学习的尺度间适配)
        # 用 1×1 卷积让网络学习如何分配信息到各尺度
        self.pre_quant_convs = nn.ModuleList([
            nn.Conv2d(cfg.latent_dim, cfg.latent_dim, 1)
            for _ in self.scales
        ])

    def encode(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        多尺度量化编码
        
        Args:
            z: (B, D, H_lat, W_lat) encoder 输出的连续 latent
        Returns:
            z_q:           (B, D, H_lat, W_lat) 量化后的完整 latent (所有尺度累加)
            z_q_scales:    List[Tensor] 每个尺度的量化结果 (在原始分辨率)
            indices_scales: List[Tensor(B, r*r)] 每个尺度的 codebook 索引
            total_commit_loss: scalar 总 commitment loss
        """
        B, D, H, W = z.shape
        
        total_commit_loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        z_q_accumulated = torch.zeros_like(z)  # 已量化的累积
        z_q_scales = []
        indices_scales = []

        residual = z  # 初始残差 = 完整 latent

        for i, (scale, quantizer, pre_conv) in enumerate(
            zip(self.scales, self.quantizers, self.pre_quant_convs)
        ):
            # 1) 自适应池化到目标分辨率
            if scale < H:
                z_pooled = F.adaptive_avg_pool2d(residual, (scale, scale))
            else:
                z_pooled = residual  # 最细尺度 = 原始分辨率

            # 2) 尺度适配
            z_pooled = pre_conv(z_pooled)

            # 3) 向量量化
            z_q_s, indices_s, commit_loss_s = quantizer(z_pooled)

            # 4) 上采样回原始分辨率
            if scale < H:
                z_q_s_up = F.interpolate(
                    z_q_s, size=(H, W), mode='bilinear', align_corners=False
                )
            else:
                z_q_s_up = z_q_s

            # 5) 累加
            z_q_accumulated = z_q_accumulated + z_q_s_up

            # 6) 更新残差: 下一个尺度量化的是当前残差减去已量化的部分
            residual = z - z_q_accumulated

            z_q_scales.append(z_q_s_up)
            indices_scales.append(indices_s)
            total_commit_loss = total_commit_loss + commit_loss_s

        # 平均 loss
        total_commit_loss = total_commit_loss / len(self.scales)

        return z_q_accumulated, z_q_scales, indices_scales, total_commit_loss

    def decode_from_indices(
        self, indices_scales: List[torch.Tensor], target_size: int
    ) -> torch.Tensor:
        """
        从 codebook 索引重建量化 latent
        
        Args:
            indices_scales: List[Tensor(B, r*r)] 各尺度的索引
            target_size: 目标空间分辨率 (H_lat)
        Returns:
            z_q: (B, D, target_size, target_size)
        """
        B = indices_scales[0].shape[0]
        device = indices_scales[0].device
        z_q = torch.zeros(B, self.latent_dim, target_size, target_size, device=device)

        for i, (scale, quantizer, indices) in enumerate(
            zip(self.scales, self.quantizers, indices_scales)
        ):
            # indices: (B, r*r) → 查表
            z_q_flat = quantizer.embedding(indices)  # (B, r*r, D)
            z_q_s = z_q_flat.reshape(B, scale, scale, self.latent_dim)
            z_q_s = z_q_s.permute(0, 3, 1, 2)  # (B, D, r, r)

            # pre_quant_conv 的逆不存在，但 straight-through 让 z_q_s 近似
            # 上采样
            if scale < target_size:
                z_q_s = F.interpolate(
                    z_q_s, size=(target_size, target_size),
                    mode='bilinear', align_corners=False
                )

            z_q = z_q + z_q_s

        return z_q

    def get_codebook_usage(self) -> List[float]:
        """统计各尺度 codebook 的使用率 (用于监控 codebook collapse)"""
        usages = []
        for quantizer in self.quantizers:
            used = (quantizer.ema_cluster_size > 1.0).float().mean().item()
            usages.append(used)
        return usages


# ============================================================
#  Multi-Scale VQVAE 完整模型
# ============================================================

class MultiScaleVQVAE(nn.Module):
    """
    多尺度向量量化 VAE
    
    完整流程:
    1. Encoder: x → z (连续 latent)
    2. MultiScaleQuantizer: z → z_q + indices (多尺度离散化)
    3. Decoder: z_q → x_recon (重建)
    
    训练损失:
    - L_recon: ||x - x_recon||^2 (重建损失)
    - L_commit: commitment loss (encoder 输出接近 codebook)
    - L_perceptual (可选): 感知损失
    
    关键属性:
    - latent_size: encoder 输出的空间尺寸 (如 8×8)
    - latent_dim: 每个位置的通道数
    - multi_scales: 量化尺度列表 (如 [1, 2, 4, 8])
    
    对接 Diffusion:
    - 训练好 VQVAE 后，冻结其参数
    - Diffusion 在 z 或 z_q 空间上操作 (latent diffusion)
    - 用 encode() 得到 latent，用 decode() 从 latent 还原图像
    """

    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.quantizer = MultiScaleQuantizer(cfg)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        编码 + 多尺度量化
        
        Args:
            x: (B, C, H, W) 输入图像, 值域 [-1, 1]
        Returns:
            z:             (B, D, H_lat, W_lat) 连续 latent (量化前)
            z_q:           (B, D, H_lat, W_lat) 量化 latent (所有尺度累加)
            z_q_scales:    List[Tensor] 每个尺度的量化结果
            indices_scales: List[Tensor] 每个尺度的 codebook 索引
            commit_loss:   scalar
        """
        z = self.encoder(x)
        z_q, z_q_scales, indices_scales, commit_loss = self.quantizer.encode(z)
        return z, z_q, z_q_scales, indices_scales, commit_loss

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        解码量化 latent → 图像
        
        Args:
            z_q: (B, D, H_lat, W_lat)
        Returns:
            x_recon: (B, C, H, W) 值域接近 [-1, 1]
        """
        return self.decoder(z_q)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        完整前向传播
        
        Args:
            x: (B, C, H, W)
        Returns:
            x_recon:       (B, C, H, W) 重建图像
            z:             (B, D, H_lat, W_lat) 连续 latent
            z_q_scales:    List[Tensor] 各尺度量化结果
            commit_loss:   scalar
        """
        z, z_q, z_q_scales, indices_scales, commit_loss = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, z, z_q_scales, commit_loss

    def compute_loss(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算完整训练损失
        
        Args:
            x: (B, C, H, W)
        Returns:
            loss: scalar 总损失
            log_dict: dict 各项损失的标量值 (用于日志)
        """
        x_recon, z, z_q_scales, commit_loss = self.forward(x)

        # 重建损失
        recon_loss = F.mse_loss(x_recon, x)

        # 总损失
        loss = recon_loss + self.cfg.commitment_weight * commit_loss

        # 日志
        log_dict = {
            'loss_total': loss.item(),
            'loss_recon': recon_loss.item(),
            'loss_commit': commit_loss.item(),
            'codebook_usage': self.quantizer.get_codebook_usage(),
        }

        return loss, log_dict

    @torch.no_grad()
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取连续 latent (用于 diffusion 训练, VQVAE 已冻结)
        
        返回量化前的连续 z，而非量化后的 z_q
        这样 diffusion 可以学习连续分布，推理时再量化
        """
        z = self.encoder(x)
        return z

    @torch.no_grad()
    def get_quantized_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        获取量化后的 latent 和各尺度表示
        """
        z, z_q, z_q_scales, indices_scales, _ = self.encode(x)
        return z_q, z_q_scales

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码连续 latent (diffusion 采样后调用)
        先量化再解码，保证离散性
        """
        z_q, _, _, _ = self.quantizer.encode(z)
        return self.decoder(z_q)

    @torch.no_grad()
    def decode_continuous(self, z: torch.Tensor) -> torch.Tensor:
        """
        直接解码连续 latent，不经过量化
        当 diffusion 在连续 z 空间上操作时，直接解码更好
        (量化会引入额外噪声)
        """
        return self.decoder(z)


# ============================================================
#  向量量化的频率分解可视化辅助
# ============================================================

class FrequencyDecomposer:
    """
    利用多尺度量化实现的频率分解
    
    基于 multi_scales = [1, 2, 4, 8] 的 VQVAE:
    - 低频 = scale 1 + scale 2 的上采样累加
    - 中频 = scale 4 的上采样  
    - 高频 = scale 8 (最细尺度的残差)
    
    这替代了之前的 SVD 截断:
    - SVD: 对图像做奇异值分解 → top-k singular values = 低频
    - Multi-Scale VQ: 粗尺度 codebook 自然捕获低频结构
    """

    @staticmethod
    @torch.no_grad()
    def decompose(
        vqvae: MultiScaleVQVAE, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        将图像分解为低频、中频、高频
        
        Returns:
            low_freq:  (B, C, H, W) 低频重建 (scale 1 + 2)
            mid_freq:  (B, C, H, W) 中频重建 (scale 4)
            high_freq: (B, C, H, W) 高频重建 (scale 8)
            all_scales: List[Tensor(B, C, H, W)] 各尺度独立重建
        """
        z, z_q, z_q_scales, indices_scales, _ = vqvae.encode(x)

        all_scales = []
        for z_q_s in z_q_scales:
            x_s = vqvae.decode(z_q_s)
            all_scales.append(x_s)

        num_scales = len(z_q_scales)

        # 低频: 前半尺度
        low_z = sum(z_q_scales[:num_scales // 2])
        low_freq = vqvae.decode(low_z)

        # 中频: 中间尺度
        mid_start = num_scales // 2
        mid_end = mid_start + max(1, (num_scales - mid_start) // 2)
        mid_z = sum(z_q_scales[mid_start:mid_end])
        mid_freq = vqvae.decode(mid_z)

        # 高频: 后半尺度
        high_z = sum(z_q_scales[mid_end:])
        high_freq = vqvae.decode(high_z)

        return low_freq, mid_freq, high_freq, all_scales


# ============================================================
#  工厂函数
# ============================================================

def VQVAE_Small(image_size: int = 32, **kwargs) -> MultiScaleVQVAE:
    """小型 VQVAE (约 5M 参数, 适合 CIFAR-10)"""
    defaults = dict(
        image_size=image_size,
        hidden_dim=128,
        latent_dim=32,
        num_res_blocks=2,
        ch_mult=[1, 2, 4],
        multi_scales=[1, 2, 4, 8],
        codebook_size=512,
    )
    defaults.update(kwargs)
    cfg = VQVAEConfig(**defaults)
    return MultiScaleVQVAE(cfg)


def VQVAE_Base(image_size: int = 32, **kwargs) -> MultiScaleVQVAE:
    """基础 VQVAE (约 14M 参数)"""
    defaults = dict(
        image_size=image_size,
        hidden_dim=256,
        latent_dim=32,
        num_res_blocks=2,
        ch_mult=[1, 2, 4],
        multi_scales=[1, 2, 4, 8],
        codebook_size=1024,
    )
    defaults.update(kwargs)
    cfg = VQVAEConfig(**defaults)
    return MultiScaleVQVAE(cfg)


def VQVAE_Large(image_size: int = 64, **kwargs) -> MultiScaleVQVAE:
    """大型 VQVAE (约 45M 参数, 适合 CelebA-HQ/LSUN)"""
    defaults = dict(
        image_size=image_size,
        hidden_dim=256,
        latent_dim=64,
        num_res_blocks=3,
        ch_mult=[1, 2, 4, 8],
        multi_scales=[1, 2, 4, 8, 16],
        codebook_size=2048,
    )
    defaults.update(kwargs)
    cfg = VQVAEConfig(**defaults)
    return MultiScaleVQVAE(cfg)

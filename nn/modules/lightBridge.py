import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = "LightweightDomainAlignBridge"

class FocalModulationBlock(nn.Module):
    """
    轻量Focal Modulation块（基于CVPR 2023 Focal Modulation Networks）
    核心：用深度卷积提取局部特征，逐点卷积生成调制权重，实现高效特征交互
    """
    def __init__(self, dim, kernel_size=3, reduction=4):
        super().__init__()
        self.dim = dim  # 输入通道数
        self.kernel_size = kernel_size  # 局部特征提取核大小
        self.dim_reduced = dim // reduction  # 调制权重压缩维度（平衡性能与参数量）

        # 1. 局部特征提取（深度卷积，参数量：dim * kernel_size^2）
        self.local_conv = nn.Conv2d(
            dim, dim, kernel_size, padding=kernel_size//2, groups=dim, bias=False
        )
        # 2. 调制权重生成（压缩→激活→扩张，参数量：dim*(dim/reduction) + (dim/reduction)*dim）
        self.compress_conv = nn.Conv2d(dim, self.dim_reduced, 1, bias=False)
        self.act = nn.GELU()  # 2023年主流激活函数，比ReLU更优
        self.expand_conv = nn.Conv2d(self.dim_reduced, dim, 1, bias=False)
        # 3. LayerScale（ICLR 2023 ConvNeXt V2）：通道级特征缩放（参数量：dim）
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1))
        # 4. 残差连接归一化（轻量BN，参数量：2*dim）
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x  # 残差保存原始特征
        # 局部特征提取
        local_feat = self.local_conv(x)
        # 调制权重生成：压缩维度→激活→恢复维度
        mod_weight = self.compress_conv(local_feat)
        mod_weight = self.act(mod_weight)
        mod_weight = self.expand_conv(mod_weight)
        # 调制操作：局部特征 × 全局调制权重（捕捉跨域共性）
        modulated_feat = local_feat * mod_weight
        # LayerScale + 残差连接（增强特征鲁棒性）
        out = self.layer_scale * modulated_feat + self.bn(residual)
        return out


class LightweightDomainAlignBridge(nn.Module):
    """
    轻量化域对齐桥阶层（YOLOv8即插即用）
    输入：YOLOv8中间特征层 (b, c, h, w)
    输出：域对齐后特征层 (b, c, h, w)
    """
    def __init__(self, in_channels=128, out_channels=None, num_blocks=1):
        super().__init__()
        # 确保输入输出通道一致（即插即用要求）
        self.out_channels = out_channels if out_channels is not None else in_channels
        assert in_channels == self.out_channels, "Input/Output channels must match for YOLOv8 plug-and-play"

        # 核心：堆叠Focal Modulation块（1~2块足够，参数量可控）
        self.align_blocks = nn.Sequential(
            *[FocalModulationBlock(dim=in_channels) for _ in range(num_blocks)]
        )

        # 域自适应权重（1个参数，动态调整对齐强度，对抗训练中优化）
        self.domain_align_weight = nn.Parameter(torch.tensor([0.5]))
        # 最终归一化（稳定训练，提升域鲁棒性）
        self.final_bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x_f):
        # x_f: YOLOv8中间特征 (b, c, h, w)
        # 1. 特征对齐
        aligned_feat = self.align_blocks(x_f)
        # 2. 域自适应融合：原始特征 × (1-权重) + 对齐特征 × 权重（平衡原始信息与对齐信息）
        out = (1 - self.domain_align_weight) * x_f + self.domain_align_weight * aligned_feat
        # 3. 最终归一化（稳定域对抗训练）
        out = self.final_bn(out)
        return out
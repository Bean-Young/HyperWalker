"""
FiLM-based Image-EHR Fusion Module
使用FiLM (Feature-wise Linear Modulation) 进行图像和EHR的融合
改进版本：添加残差连接、恒等初始化、L2归一化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMFusion(nn.Module):
    """
    FiLM融合模块（改进版）
    使用EHR特征对图像特征进行缩放和偏移
    公式: γ, β = MLP(e), f = Norm(Original_Image + (γ ⊙ v + β))
    
    改进：
    1. 残差连接：确保输出不会偏离原始图像特征太远
    2. 恒等初始化：训练初期γ=0, β=0，保证恒等变换
    3. L2归一化：确保输出在单位超球面上
    """
    def __init__(self, dim: int = 1024):
        """
        Args:
            dim: 特征维度（保持不变）
        """
        super().__init__()
        self.dim = dim
        
        # MLP用于从EHR embedding生成γ和β
        # 输入: EHR pooled embedding (dim,)
        # 输出: γ和β各为 (dim,)
        self.film_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),  # 输出2*dim，分别用于γ和β
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2)  # 最终输出2*dim
        )
        
        # LayerNorm用于残差连接后的归一化
        self.norm = nn.LayerNorm(dim)
        
        # 初始化策略：让最后一层从恒等变换开始
        # 当γ=0, β=0时，f = γ ⊙ v + β = 0，残差连接后 f = v + 0 = v（恒等变换）
        self._init_identity()
    
    def _init_identity(self):
        """
        初始化策略：让模型从恒等变换开始
        将film_mlp最后一层的weight初始化为0，bias初始化为0
        这样在训练初期，γ=0, β=0，输出 = Norm(Original_Image + 0) = Norm(Original_Image)
        """
        # 获取最后一层（第二个Linear层）
        last_layer = self.film_mlp[-1]
        if isinstance(last_layer, nn.Linear):
            # weight初始化为0
            nn.init.zeros_(last_layer.weight)
            # bias初始化为0
            nn.init.zeros_(last_layer.bias)
    
    def forward(self, image_embeds: torch.Tensor, ehr_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: (B, dim) 图像pooled embedding（已经pooling过）
            ehr_embeds: (B, dim) EHR pooled embedding（已经pooling过）
        Returns:
            fused: (B, dim) 融合后的embedding（L2归一化）
        """
        # 输入应该已经是pooled的embedding (B, dim)
        # 如果输入是 (B, seq_len, dim)，则先pooling
        if len(image_embeds.shape) == 3:
            image_embeds = image_embeds.mean(dim=1)  # (B, dim)
        if len(ehr_embeds.shape) == 3:
            ehr_embeds = ehr_embeds.mean(dim=1)  # (B, dim)
        
        # 保存原始图像embedding用于残差连接
        original_image = image_embeds
        
        # 从EHR生成γ和β
        film_params = self.film_mlp(ehr_embeds)  # (B, 2*dim)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # 各为 (B, dim)
        
        # FiLM变换: mlp_output = γ ⊙ v + β
        mlp_output = gamma * image_embeds + beta  # (B, dim)
        
        # 残差连接：fused = Original_Image + MLP_Output
        # 这样即使MLP输出很差，至少还能保留原始图像特征
        fused = original_image + mlp_output  # (B, dim)
        
        # LayerNorm（残差连接后的归一化）
        fused = self.norm(fused)  # (B, dim)
        
        # L2归一化：确保输出在单位超球面上，与超图中的节点embedding在同一空间
        fused = F.normalize(fused, p=2, dim=-1)  # (B, dim)
        
        return fused


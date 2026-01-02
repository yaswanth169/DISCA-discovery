"""3D CNN Feature Extractor for Subtomogram Volumes."""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D convolutional block with BatchNorm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=not use_batch_norm,
        )
        self.batch_norm = nn.BatchNorm3d(out_channels) if use_batch_norm else None
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class FeatureExtractor3D(nn.Module):
    """3D CNN for extracting features from subtomogram volumes."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.input_size = config.get("model", {}).get("input_size", [32, 32, 32])
        feature_config = config.get("model", {}).get("feature_extractor", {})
        
        self.conv_channels = feature_config.get("conv_channels", [32, 64, 128, 256])
        self.kernel_sizes = feature_config.get("kernel_sizes", [3, 3, 3, 3])
        self.strides = feature_config.get("strides", [1, 2, 2, 2])
        self.padding = feature_config.get("padding", [1, 1, 1, 1])
        self.use_batch_norm = feature_config.get("use_batch_norm", True)
        self.dropout_rate = feature_config.get("dropout_rate", 0.1)
        self.activation = config.get("model", {}).get("activation", "relu")
        self.feature_dim = config.get("model", {}).get("feature_dim", 128)
        
        self.conv_blocks = nn.ModuleList()
        in_channels = 1
        
        for i, out_channels in enumerate(self.conv_channels):
            block = Conv3DBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                padding=self.padding[i],
                use_batch_norm=self.use_batch_norm,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(self.conv_channels[-1], self.feature_dim)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        intermediate_features = []
        
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            if return_intermediate:
                intermediate_features.append(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        x = F.normalize(x, p=2, dim=1)
        
        if return_intermediate:
            return {"features": x, "intermediate": intermediate_features}
        return x
    
    def get_feature_dim(self) -> int:
        return self.feature_dim

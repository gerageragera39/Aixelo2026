import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class BottleneckModelConfig:
    seed: int
    depth: int
    sd_on: bool
    final_survival_prob: float
    epochs: int
    batch_size: int
    output_dir: str
    test_split: float = 0.1

    def __post_init__(self):
        valid_bottleneck_depths = [50, 101]
        
        if self.depth not in valid_bottleneck_depths:
            raise ValueError(f"Depth must be one of {valid_bottleneck_depths} for bottleneck architecture")


class BottleneckBlock(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 survival_prob: float = 1.0, downsample=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.survival_prob = survival_prob
        self.expanded_channels = out_channels * self.expansion
        self.downsample = downsample

        # 1x1 conv to reduce dimension
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # 3x3 conv with stride
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        # 1x1 conv to expand dimension
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, self.expanded_channels, kernel_size=1, bias=False)

    
    # def forward(self, x):
    #     identity = x
        
    #     out = self.bn1(x)
    #     out = self.relu1(out)
    #     out = self.conv1(out)

    #     out = self.bn2(out)
    #     out = self.relu2(out)
    #     out = self.conv2(out)

    #     out = self.bn3(out)
    #     out = self.relu3(out)
    #     out = self.conv3(out)

    #     if self.downsample is not None:
    #         identity = self.downsample(x)

    #     if self.training and self.survival_prob < 1.0:
    #         mask_shape = (x.shape[0], 1, 1, 1)
    #         mask = torch.empty(mask_shape, device=x.device).bernoulli_(self.survival_prob)
            
    #         out = (out / self.survival_prob) * mask
            
    #     return identity + out

    def forward(self, x):
        identity = x

        def residual(x):
            out = self.bn1(x)
            out = self.relu1(out)
            out = self.conv1(out)

            out = self.bn2(out)
            out = self.relu2(out)
            out = self.conv2(out)

            out = self.bn3(out)
            out = self.relu3(out)
            out = self.conv3(out)
            return out

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.training and self.survival_prob < 1.0:
            if torch.rand(1, device=x.device) < self.survival_prob:
                out = residual(x) / self.survival_prob
                return identity + out
            else:
                return identity
        else:
            out = residual(x)
            return identity + out



class ResNetBottleneck(nn.Module):
    def __init__(self, config: BottleneckModelConfig, num_classes: int = 102):
        super().__init__()
        self.num_classes = num_classes
        self.sd_on = config.sd_on
        self.final_survival_prob = config.final_survival_prob

        # Define number of blocks per stage for different ResNet depths
        if config.depth == 50:
            self.layers_config = [3, 4, 6, 3] 
        elif config.depth == 101:
            self.layers_config = [3, 4, 23, 3]  
        else:
            raise ValueError(f"Unsupported depth: {config.depth}")

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        total_blocks = sum(self.layers_config)
        block_idx = 0

        # Create stages
        self.stage1 = self._make_stage(64, 64, self.layers_config[0], stride=1, block_idx=block_idx, total_blocks=total_blocks)
        block_idx += self.layers_config[0]
        self.stage2 = self._make_stage(256, 128, self.layers_config[1], stride=2, block_idx=block_idx, total_blocks=total_blocks)
        block_idx += self.layers_config[1]
        self.stage3 = self._make_stage(512, 256, self.layers_config[2], stride=2, block_idx=block_idx, total_blocks=total_blocks)
        block_idx += self.layers_config[2]
        self.stage4 = self._make_stage(1024, 512, self.layers_config[3], stride=2, block_idx=block_idx, total_blocks=total_blocks)

        # Final batch norm and classifier
        self.bn_final = nn.BatchNorm2d(2048)  # 512 * expansion (4)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, block_idx, total_blocks):
        layers = []
        downsample = None
        
        # Check if we need to downsample in the first block of the stage
        if stride != 1 or in_channels != out_channels * BottleneckBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleneckBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleneckBlock.expansion)
            )

        for i in range(num_blocks):
            if self.sd_on:
                current_l = block_idx + i
                survival_prob = 1 - (current_l / (total_blocks - 1)) * (1 - self.final_survival_prob)
            else:
                survival_prob = 1.0

            if i == 0:
                layers.append(BottleneckBlock(in_channels, out_channels, stride, survival_prob, downsample))
                in_channels = out_channels * BottleneckBlock.expansion
            else:
                layers.append(BottleneckBlock(in_channels, out_channels, 1, survival_prob))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.bn_final(x)
        x = self.relu_final(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

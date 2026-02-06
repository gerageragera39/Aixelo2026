import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    seed: int
    depth: int
    sd_on: bool
    final_survival_prob: float
    epochs: int
    batch_size: int
    output_dir: str
    test_split: float = 0.1

    def __post_init__(self):
        assert (self.depth - 2) % 6 == 0, "Depth must be 20, 32, 44, 56, 110"


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, survival_prob: float):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.survival_prob = survival_prob

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                              padding=1, bias=False, padding_mode='zeros')

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1, bias=False, padding_mode='zeros')

        #shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                         bias=False, padding_mode='zeros'),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

# '''Sample Mode Stochastic Depth'''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.training and self.survival_prob < 1.0:
            shape = (x.shape[0], 1, 1, 1)
            mask = torch.empty(shape, device=x.device).bernoulli_(self.survival_prob)
            out = (out / self.survival_prob) * mask
            
        return shortcut + out


class ResNetBlockBatchDrop(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, survival_prob: float):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.survival_prob = survival_prob

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                              padding=1, bias=False, padding_mode='zeros')

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1, bias=False, padding_mode='zeros')

        #shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                         bias=False, padding_mode='zeros'),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    # '''Batch Mode Stochastic Depth'''
    def forward(self, x):
        shortcut = self.shortcut(x)

        if self.training and self.survival_prob < 1.0:
            if torch.rand(1, device=x.device) < self.survival_prob:
                out = self.bn1(x)
                out = self.relu1(out)
                out = self.conv1(out)

                out = self.bn2(out)
                out = self.relu2(out)
                out = self.conv2(out)

                out = shortcut + out / self.survival_prob
            else:
                out = shortcut
        else:
            out = self.bn1(x)
            out = self.relu1(out)
            out = self.conv1(out)

            out = self.bn2(out)
            out = self.relu2(out)
            out = self.conv2(out)

            out = shortcut + out

        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, config: ModelConfig, num_classes: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.sd_on = config.sd_on
        self.final_survival_prob = config.final_survival_prob

        n = (config.depth - 2) // 6  # Number of blocks per stage
        total_blocks = 3 * n  # 3 stages
        block_idx = 0

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False,
                              padding_mode='zeros')
        filters_list = [16, 32, 64]
        layers = []
        in_channels = 16

        for stage, out_channels in enumerate(filters_list):
            for i in range(n):
                stride = 1 if i == 0 and stage == 0 else 2 if i == 0 else 1

                if config.sd_on:
                    survival_prob = 1 - (block_idx / (total_blocks - 1)) * (1 - config.final_survival_prob)
                    block_idx += 1
                else:
                    survival_prob = 1.0

                layers.append(ResNetBlockBatchDrop(in_channels, out_channels, stride, survival_prob))
                in_channels = out_channels

        self.layers = nn.Sequential(*layers)

        # Final batch norm and global average pooling
        self.bn_final = nn.BatchNorm2d(out_channels)
        self.relu_final = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        x = self.layers(x)

        x = self.bn_final(x)
        x = self.relu_final(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


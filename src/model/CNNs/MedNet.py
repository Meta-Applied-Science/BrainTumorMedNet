import torch
from torch import nn

class MedNet(nn.Module):
    """Implements the MedNet CNN architecture:contentReference[oaicite:17]{index=17}:contentReference[oaicite:18]{index=18}.
    
    MedNet is a deep convolutional neural network designed for medical image classification,
    consisting of a total of 44 convolutional layers:contentReference[oaicite:19]{index=19}. 
    The network accepts input images with 1 (grayscale) or 3 (RGB) channels of size 224x224, 
    and outputs class probabilities over a specified number of classes (default 4).
    
    Architecture:
      - Two initial convolutional layers applied sequentially (3×3 then 5×5 filters), each followed by 
        batch normalization and ReLU activation:contentReference[oaicite:20]{index=20}. These serve as a feature extractor 
        "stem" and reduce the input image spatial size.
      - Eight subsequent convolutional blocks. Each block contains four parallel convolutional layers 
        with filter sizes 1×1, 3×3, 5×5, and 7×7 (with appropriate padding), to capture multi-scale features:contentReference[oaicite:21]{index=21}. 
        Outputs of these four convolutions are concatenated along the channel dimension. 
        Each convolution in these blocks is followed by batch normalization and ReLU activation:contentReference[oaicite:22]{index=22}.
      - Twelve residual skip connections between the convolutional blocks (in addition to the normal sequential flow) 
        to improve feature propagation and gradient flow:contentReference[oaicite:23]{index=23}. These skips connect non-adjacent blocks 
        (some short-range, some long-range) and are implemented as 1×1 convolutions (with stride as needed) 
        to match dimensions before summation:contentReference[oaicite:24]{index=24}.
      - After the eight blocks, a global average pooling layer aggregates each feature map to a single value:contentReference[oaicite:25]{index=25}.
      - A fully-connected classifier consisting of two linear layers (with a ReLU and dropout in between) 
        and a final Softmax activation produces the output probabilities:contentReference[oaicite:26]{index=26}.
    
    Args:
        input_channels (int): Number of channels in the input images (1 for grayscale, 3 for RGB).
        num_classes (int): Number of output classes for classification. Default is 4.
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 4):
        super(MedNet, self).__init__()
        # Initial two conv layers (3x3 and 5x5) with BN and ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Define output channels for each stage of blocks
        stage_channels = [64, 128, 256, 512]  # Stage0 -> 64, Stage1 -> 128, Stage2 -> 256, Stage3 -> 512
        # Create parallel convolution blocks
        self.blocks_conv = nn.ModuleList()
        in_channels = 64  # output of conv2
        current_stage = 0
        for block_idx in range(1, 9):
            # If this block starts a new stage (block 3,5,7), advance stage index (downsample here)
            if block_idx in {3, 5, 7}:
                current_stage += 1
            out_channels_total = stage_channels[current_stage]     # total output channels for this block
            out_per_branch = out_channels_total // 4               # channels per each parallel conv branch
            stride_val = 2 if block_idx in {3, 5, 7} else 1        # downsample at blocks 3,5,7
            # Define four parallel conv branches for 1x1, 3x3, 5x5, 7x7
            branch1 = nn.Sequential(
                nn.Conv2d(in_channels, out_per_branch, kernel_size=1, stride=stride_val, padding=0),
                nn.BatchNorm2d(out_per_branch),
                nn.ReLU(inplace=True)
            )
            branch3 = nn.Sequential(
                nn.Conv2d(in_channels, out_per_branch, kernel_size=3, stride=stride_val, padding=1),
                nn.BatchNorm2d(out_per_branch),
                nn.ReLU(inplace=True)
            )
            branch5 = nn.Sequential(
                nn.Conv2d(in_channels, out_per_branch, kernel_size=5, stride=stride_val, padding=2),
                nn.BatchNorm2d(out_per_branch),
                nn.ReLU(inplace=True)
            )
            branch7 = nn.Sequential(
                nn.Conv2d(in_channels, out_per_branch, kernel_size=7, stride=stride_val, padding=3),
                nn.BatchNorm2d(out_per_branch),
                nn.ReLU(inplace=True)
            )
            self.blocks_conv.append(nn.ModuleList([branch1, branch3, branch5, branch7]))
            # Update in_channels for the next block (equal to this block's output channels)
            in_channels = out_channels_total
        # Define skip connection conv layers (ModuleDict for easy lookup by 'j->i' key)
        self.skip_convs = nn.ModuleDict()
        # Map each block index to its stage index for determining channel counts and stride
        block_stage = {i: (i-1)//2 for i in range(1, 9)}
        for j in range(1, 9):
            for i in range(j+1, 9):
                if (j % 2) == (i % 2):  # skip connections only between blocks of same parity (odd->odd or even->even)
                    stage_j = block_stage[j]
                    stage_i = block_stage[i]
                    in_ch = stage_channels[stage_j]
                    out_ch = stage_channels[stage_i]
                    # Determine stride for spatial alignment (downsample if connecting to a deeper stage)
                    stride_val = 1
                    if stage_i > stage_j:
                        stride_val = 2 ** (stage_i - stage_j)
                    # 1x1 conv for skip from block j to block i
                    self.skip_convs[f"{j}->{i}"] = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride_val, padding=0)
        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # Fully connected classifier (two linear layers with dropout and ReLU)
        hidden_units = stage_channels[-1] // 2  # half of last conv channels, e.g., 512 -> 256
        self.fc1 = nn.Linear(stage_channels[-1], hidden_units)
        self.fc_relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        # Final softmax layer for output probabilities
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        # Dictionary to store outputs of each block (for skip connections)
        block_outputs = {}
        # Iterate through each of the 8 blocks
        for idx, branches in enumerate(self.blocks_conv, start=1):
            # Add skip connection contributions from earlier blocks of same parity
            for j in range(1, idx):
                if (j % 2) == (idx % 2):
                    skip_key = f"{j}->{idx}"
                    if skip_key in self.skip_convs:
                        x = x + self.skip_convs[skip_key](block_outputs[j])
            # Compute parallel conv outputs for current block
            branch_outs = [branch(x) for branch in branches]    # list of outputs from 4 branches
            concat_out = torch.cat(branch_outs, dim=1)         # concatenate along channel dimension
            block_outputs[idx] = concat_out                    # store output for future skips
            x = concat_out                                     # input to next block
        # Global average pooling to 1×1
        x = self.global_pool(x)        # shape: (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)      # flatten to (batch, channels)
        # Fully connected layers with dropout
        x = self.fc1(x)
        x = self.fc_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # Softmax to get class probability outputs
        x = self.softmax(x)
        return x


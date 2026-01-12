import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block providing channel-wise attention.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UpsampleResBlock(nn.Module):
    """
    Upsampling block with a residual connection.
    """

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_conv = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.upsample_conv(x)
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(self.bn(out))
        return out


class EnhancedResNetDecoder(nn.Module):
    """
    Residual decoder composed of upsampling blocks and channel attention.
    """

    def __init__(self, input_channels=256, upsample_stages=5):
        super().__init__()

        num_blocks_before_attention = upsample_stages - 2
        if not (1 <= num_blocks_before_attention <= 3):
            raise ValueError(f"upsample_stages must be 3, 4, or 5. Received: {upsample_stages}")

        self.blocks_before = nn.ModuleList()
        current_channels = input_channels

        if num_blocks_before_attention == 3:  # upsample_stages = 5
            self.blocks_before.append(UpsampleResBlock(current_channels, 128, scale_factor=2))
            self.blocks_before.append(UpsampleResBlock(128, 64, scale_factor=2))
            self.blocks_before.append(UpsampleResBlock(64, 64, scale_factor=2))
        elif num_blocks_before_attention == 2:  # upsample_stages = 4
            self.blocks_before.append(UpsampleResBlock(current_channels, 64, scale_factor=4))
            self.blocks_before.append(UpsampleResBlock(64, 64, scale_factor=2))
        elif num_blocks_before_attention == 1:  # upsample_stages = 3
            self.blocks_before.append(UpsampleResBlock(current_channels, 64, scale_factor=8))

        self.attention = SEBlock(64)

        self.block4 = UpsampleResBlock(64, 32, scale_factor=2)  # 56x56 -> 112x112
        self.block5 = UpsampleResBlock(32, 16, scale_factor=2)  # 112x112 -> 224x224

        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x, return_pre_activation=True):
        for block in self.blocks_before:
            x = block(x)
        x = self.attention(x)
        x = self.block4(x)
        x = self.block5(x)

        logits = self.final_conv(x)
        if return_pre_activation:
            return logits, logits
        return logits, None


class SaliencyDecoder(nn.Module):
    def __init__(self, input_dim, output_dim_flat=256 * 7 * 7, num_models=256, upsample_stages=5):
        super().__init__()
        if output_dim_flat % num_models != 0:
            raise ValueError(f"output_dim_flat ({output_dim_flat}) must be divisible by num_models ({num_models})")

        # single_model_output_dim = output_dim_flat // num_models
        # self.models = nn.ModuleList([nn.Linear(input_dim, single_model_output_dim) for _ in range(num_models)])
        self.linear = nn.Linear(input_dim, output_dim_flat)

        self.decoder_channels = 256
        self.decoder_H = 7
        self.decoder_W = 7

        if self.decoder_channels * self.decoder_H * self.decoder_W != output_dim_flat:
            raise ValueError("Inconsistency between output_dim_flat and reshape dimensions.")

        self.decoder = EnhancedResNetDecoder(
            input_channels=self.decoder_channels,
            upsample_stages=upsample_stages,
        )

    def forward(self, x, return_pre_activation=True):
        # outputs = [model(x) for model in self.models]
        # outputs_cat = torch.cat(outputs, dim=1)
        outputs_cat = self.linear(x)
        
        reshaped_for_decoder = outputs_cat.view(x.shape[0], self.decoder_channels, self.decoder_H, self.decoder_W)

        decoded_output_logits, pre_activation = self.decoder(
            reshaped_for_decoder, return_pre_activation=return_pre_activation
        )
        B, C, H, W = decoded_output_logits.shape
        decoded_output_logits = decoded_output_logits.view(B, C, -1)
        log_probs = F.log_softmax(decoded_output_logits, dim=2).view(B, C, H, W)

        return reshaped_for_decoder, (pre_activation, log_probs)
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Union, Sequence
from .common import LayerNorm2d

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm2d(F_int)
        )


        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm2d(F_int)
        )


        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_size = x.size()[2:]
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        g1_aligned = F.interpolate(g1, size=x_size, mode='bilinear', align_corners=False)
        psi = self.relu(g1_aligned + x1)
        psi = self.psi(psi)

        return x * psi


class FusionBlock(nn.Module):
    def __init__(
            self, in_channels_skip: int, in_channels_prev: int, out_channels: int,
            activation: Type[nn.Module] = nn.GELU
    ):
        super().__init__()
        self.att_gate = AttentionGate(F_g=in_channels_prev, F_l=in_channels_skip, F_int=in_channels_skip // 2)

        self.conv1 = nn.Conv2d(in_channels_skip + in_channels_prev, out_channels, kernel_size=3, padding=1, bias=False)
        # self.conv1 = DynamicDilatedConv(in_channels_skip + in_channels_prev, out_channels, kernel_size=3, padding=None)
        self.norm1 = LayerNorm2d(out_channels)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # self.conv2 = DynamicDilatedConv(out_channels, out_channels, kernel_size=3, padding=None)
        self.norm2 = LayerNorm2d(out_channels)
        self.act2 = activation()


        self.attention = CBAM(out_channels)

        self.se_block = SEBlock(out_channels)


    def forward(self, prev_features: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:

        gated_skip_features = self.att_gate(g=prev_features, x=skip_features)
        combined_features = torch.cat([prev_features, gated_skip_features], dim=1)

        x = self.conv1(combined_features)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        out = self.attention(x)

        out = self.se_block(out)

        return out

class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            unet_mode: bool = False,
            encoder_embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.unet_mode = unet_mode

        if self.unet_mode:

            fusion_channels = (512, 256, 128)
            encoder_channels = [encoder_embed_dim] * 3 + [transformer_dim]
            encoder_channels = encoder_channels[::-1]

            self.fusion_blocks = nn.ModuleList()

            self.fusion_blocks.append(
                FusionBlock(encoder_channels[1], transformer_dim, fusion_channels[0], activation)
            )
            for i in range(len(fusion_channels) - 1):
                self.fusion_blocks.append(
                    FusionBlock(encoder_channels[i + 2], fusion_channels[i], fusion_channels[i + 1], activation)
                )
            self.num_mask_tokens = num_multimask_outputs + 1
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(fusion_channels[-1], fusion_channels[-1] // 2, kernel_size=2, stride=2),
                LayerNorm2d(fusion_channels[-1] // 2),
                activation(),
                nn.ConvTranspose2d(fusion_channels[-1] // 2, fusion_channels[-1] // 4, kernel_size=2, stride=2),
                LayerNorm2d(fusion_channels[-1] // 4),
                activation(),
            )
            self.output_head = nn.Conv2d(fusion_channels[-1] // 4, self.num_mask_tokens, kernel_size=1)

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.unet_mode:
            assert isinstance(image_embeddings, list), "U-Net mode requires a list of features from the encoder."
            masks, iou_pred = self.predict_masks_unet(
                encoder_features=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
            )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks_unet(
            self,
            encoder_features: List[torch.Tensor],
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        expected_features = len(self.fusion_blocks) + 1
        assert len(encoder_features) == expected_features, \
            f"Expected {expected_features} features from encoder, but got {len(encoder_features)}"

        encoder_features = encoder_features[::-1]
        deepest_features = encoder_features[0]

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        hs, src = self.transformer(deepest_features, image_pe, tokens)
        iou_token_out = hs[:, 0, :]

        x = src.transpose(1, 2).view(src.shape[0], self.transformer_dim, 64, 64)

        for i, fusion_block in enumerate(self.fusion_blocks):
            skip_features = encoder_features[i + 1]
            x = fusion_block(x, skip_features)

        x = self.output_upscaling(x)

        masks = self.output_head(x)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.sigmoid_output = sigmoid_output

        self.use_residual = (input_dim == output_dim)
        if self.use_residual:
            print(f"(input_dim={input_dim}, output_dim={output_dim})")

        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        identity = x
        for i, layer in enumerate(self.layers):
            out = layer(x)

            if i < self.num_layers - 1:
                out = F.relu(out)
                out = self.dropout(out)

                if self.use_residual and i == 0:
                    out = out + identity
            x = out

        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
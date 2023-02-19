import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from vit_pytorch import ViT
from transformers import SegformerDecodeHead, SegformerConfig
import math


def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    This module packs `F.pixel_shuffle()` and a nn.Conv2d module together to
    achieve a simple upsampling with pixel shuffle.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio. Typically, 2 to 4.
            Depends on the problem. 2 is often enough but if the problem requires more fine grain, use 4
        upsample_kernel (int): Kernel size of the conv layer to expand the
            channels.
    """

    def __init__(self, in_channels, out_channels, upsample_kernel=3, scale_factor=4):
        super(PixelShufflePack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.upsample_conv, distribution='uniform')

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)  # channel reduce by scale_factor**2, h&w increase by scale_factor
        return x


class SegFormerDecoderMod(SegformerDecodeHead):
    def forward(self, encoder_hidden_states):
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


class ViTSeg(ViT):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=3, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim,
                         depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, channels=channels,
                         dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
        self.dropout_p = dropout
        patch_height, patch_width = pair(patch_size)

        patch_dim = channels * patch_height * patch_width
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.upsampler = PixelShufflePack(in_channels=dim, out_channels=3)
        self.linear_fuse = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(dim, eps=1e-5)
        self.linear_pred = nn.Conv2d(dim, num_classes, kernel_size=(1, 1))
        self.segformerHead = SegFormerDecoderMod(SegformerConfig())
        self.reverse_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # print('shape before self attention and transform', x.shape)  # [4, 64, 1024]
        x = self.transformer(x)  # turn x to shape [batch size, number of patches + 1 for the classes, embed_dim]
        # TODO: make the transformer layer deeper
        print('shape after transform', x.shape)  # [4, 65, 1024]



        x = x.permute(2, 0, 1)  # permute to get the number of channel at the right spot [1024, 4, 65]
        #TODO: make a for loop to go through each patch in the 65 patches, and convert them back from embedding
        #       to normal rgb. So maybe try the outchannel to be 3 in the pixel shuffle


        #TODO: how do I return the image from embedding space to rgb space
        #       must have the image in the right shape (b, c, h, w) before upsample



        # x = self.segformerHead()
        x = self.upsampler(x)
        print('shape after upsample', x.shape)  # [3, 4, 260]

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  #-> turn x into shape [batch size, embedding dim size]
        # print('shape after this weird mean func', x.shape)  #-> [4, 1024]

        # modification starts here
        # x = self.to_latent(x)
        # x = self.mlp_head(x)


        x = [x]
        x = self.linear_fuse(torch.cat(x, dim=1))
        x = self.bn(x)  #TODO: this one defintiely requires to have the right shape with the batch there
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout_p)
        x = self.linear_pred(x)
        # TODO: apply softmax after this to convert the pixels into classes

        return x  # should be the logits


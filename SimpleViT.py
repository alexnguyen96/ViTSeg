import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


class SimpleViT(nn.Module):
    def __int__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0.1):
        super.__init__()

        image_h, image_w = (image_size, image_size)
        patch_h, patch_w = (patch_size, patch_size)

        n_patch = (image_h // patch_h) * (image_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        # embedding according to ViT Torch
        self.to_patch_embedding = nn.Sequential(
            Rearrange('batch channel (height p1) (width p2) -> batch height width (p1 p2 channel)', p1=patch_h,
                      p2=image_h), # cropping the images into patches and then flatten
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head,
                                       mlp_dim)  # inside is where they do attention and feedfoward


        # embedding according to ViT vanilla
        # 1. make a CNN model with weight standardization, StdConv, groupnorm, relu, max_pool...
        # 2. ResNet?
        # 3. Make a CNN to create embeddings. Which will return a grid of embeddings
        # 4. then feed that through the encoder
        self.encoderblock = nn.Sequential(
            nn.LayerNorm(...inputs...),
            nn.MultiheadAttention(dtype=self.dtype, num_heads=dim_head, embed_dim=...), # attention block, set initializer to xavier uniform
            nn.Dropout(dropout),
            nn.LayerNorm(...),
        )
        # multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # attn_output, attn_output_weights = multihead_attn(query, key, value)
        self.encoder = nn.Sequential(
            self.addPosEmbs,
            nn.Dropout(dropout),
            self.encoderblock,
            nn.LayerNorm(...),
        )






        self.to_latent = nn.Identity()

        # classifier
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def addPosEmbs(self, inputs):
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
        return inputs + pe

    def forward(self, inputs):

        *_, h, w, dtype = *inputs.shape, inputs.dtype

        x = self.to_patch_embedding(inputs)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)


    #decoder head -> same as segformer, MLP (2D CNN 1 to 1), then 2 optiosn: upsample by steps like we discussed, or up to original size.
    # with upsample: 2 optison: mmseg CNN upsampler and bilinear

    # try the mmseg thign below first, then the decode 2, then decoder 1

    img = torch.randn(16, 1024, 16, 16) # -> ViT
    out_img = torch.randn(16, 2048, 16, 16)


class DecoderHead1(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()

        self.mlp = nn.Conv2d(2048, 256*scale_factor*scale_factor, (1, 1))
        self.bn = nn.BatchNorm2d(256*scale_factor*scale_factor, eps=1e-5)


    def forward(self, inputs):
        x = self.mlp(inputs)
        x = interpolate(x, size=(4096, 4096), mode='bilinear')
        x = self.bn(x)
        x = relu(x, inplace=True)
        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.linear_pred(x)



class DecoderHead2(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm2d(2048*8, eps=1e-5)
        self.mlp = nn.Conv2d(2048, 2048*8, (1, 1))
        self.mlp2 = nn.Conv2d(2048*8, 2048*8, (1, 1))


    def forward(self, inputs):

        x = self.mlp(inputs)
        x = interpolate(x, size=(256, 256), mode='bilinear')
        x = self.bn(x)
        x = relu(x, inplace=True)
        x = dropout(x, p=self.dropout_p, training=self.training)

        x = self.mlp2(inputs)
        x = interpolate(x, size=(4096, 4096), mode='bilinear')
        x = self.bn(x)
        x = relu(x, inplace=True)

        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.linear_pred(x)

        retrun x

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

    def __init__(self, in_channels, out_channels, upsample_kernel, scale_factor=4):
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
        x = F.pixel_shuffle(x, self.scale_factor)
        return x



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


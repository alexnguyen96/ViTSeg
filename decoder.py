import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from helper import PixelShufflePack


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Decoder(nn.Module):
    def __init__(self, patch_num, patch_size, num_classes, hidden_size, num_layers, num_heads, feedforward_size, 
                 channels=3, dim_head=64, dropout=0.1, scale_factor=4):
        self.patch_num = patch_num
        num_patches = patch_num**2
        patch_dim = channels * patch_size * patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(hidden_size, num_layers, num_heads, dim_head, feedforward_size, dropout)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # TODO: scale 4 times with scale factor 4 -> to get to 4k. Figure out the best out_channel num -> chatGPT?
        # TODO calculation to get the right img size in the end, -> find out outchanel, scalefact, kernel
        out_chan = [hidden_size*i for i in range(2, 5)]
        self.upsampler = nn.Sequential(
          PixelShufflePack(in_channels=hidden_size, out_channels=out_chan[0], scale_factor=scale_factor),
          PixelShufflePack(in_channels=out_chan[0], out_channels=out_chan[1], scale_factor=4),
          PixelShufflePack(in_channels=out_chan[1], out_channels=out_chan[2], scale_factor=2),
          PixelShufflePack(in_channels=out_chan[2], out_channels=num_classes, scale_factor=1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img):
        x = self.to_patch_embedding(img)  # shape [4, 64, 1024]
        b, n, _ = x.shape

        # positional embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # self attention
        x = self.transformer(x)  
        # TODO: make the attention layer deeper

        # remove the class token
        x = x[:, 1:, :]  # shape [4, 64, 1024]

        batch_num = x.shape[0]
        height = width = self.patch_num
        x = x.permute(0, 2, 1)  #shape [4, 1024, 64]
        x = x.reshape(batch_num, -1, height, width)  # shape [4, 1024, 8, 8]

        # print("x shape before upsample", x.shape)
        x = self.upsampler(x)
        # print("x shape AFTER upsample", x.shape)

        # classifying pixelwise
        x = self.softmax(x)
        x = x.argmax(dim=1)
        return x  # should be the logits

import torch
from torch import nn
import torch.nn.functional as F
import math

from decoder import Decoder
from encoder import SegFormerEncoder, MAEEncoder

class VitSeg(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.mae_encoder = MAEEncoder(img_size, patch_size, in_chans,
                 embed_dim, depth, num_heads,
                 decoder_embed_dim, decoder_depth, decoder_num_heads,
                 mlp_ratio, norm_layer)

        self.seg_encoder = SegFormerEncoder() #it's mit_b0 right now

        # TODO: scale 4 times with scale factor 4 -> to get to 4k. Figure out the best out_channel num -> chatGPT?
        # TODO calculation to get the right img size in the end, -> find out outchanel, scalefact, kernel
        # patch_num = image_size // patch_size
        # self.decoder = Decoder(patch_num, num_classes, hidden_size, dropout=dropout, scale_factor=4)

    def forward(self, img):
        x = self.seg_encoder(img)
        for y in x:
            print('y shape', y.shape)
            # y shape torch.Size([1, 32, 64, 64])
            # y shape torch.Size([1, 64, 32, 32])
            # y shape torch.Size([1, 160, 16, 16])
            # y shape torch.Size([1, 256, 8, 8])

        y = self.mae_encoder(img, mask_ratio=0.75)
        y = y[:, 1:, :]  # remove cls token
        # y shape [1, 64, 256] [batch, 8*8, embed_dim]

        #TODO: concat x and y -> z
        # x = z
        # x = self.decoder(x)

        return x

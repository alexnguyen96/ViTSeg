import torch
from torch import nn
import torch.nn.functional as F
import math

from decoder import Decoder
from encoder import SegFormerEncoder, MAEEncoder

class VitSeg(nn.Module):
    def __init__(self, image_size, patch_size, hidden_size, num_layers, num_heads, mlp_dim, 
                 num_classes, feedforward_size, in_channels=3, dropout=0.1):
        super().__init__()
        self.mae_encoder = MAEEncoder()

        self.seg_encoder = SegFormerEncoder() #it's mit_b0 right now

        # TODO: scale 4 times with scale factor 4 -> to get to 4k. Figure out the best out_channel num -> chatGPT?
        # TODO calculation to get the right img size in the end, -> find out outchanel, scalefact, kernel
        patch_num = image_size // patch_size
        self.decoder = Decoder(patch_num, num_classes, hidden_size, dropout=dropout, scale_factor=4)

    def forward(self, img):
        x = self.seg_encoder(img)
        for y in x:
            print('y shape', y.shape)
            # y shape torch.Size([1, 32, 128, 128])
            # y shape torch.Size([1, 64, 64, 64])
            # y shape torch.Size([1, 160, 32, 32])
            # y shape torch.Size([1, 256, 16, 16])
        #ok, for each encoder size, use the decoder to get to the same size, which is 4k


        # print("x shape after seg encoder", x.shape)
        # y = self.mae_encoder(img)
        #TODO: concat x and y -> z
        # x = z
        # x = self.decoder(x)

        return x

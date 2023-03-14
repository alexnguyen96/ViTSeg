import torch
from torch import nn
import torch.nn.functional as F
import math

from decoder import Decoder
from encoder import SegFormerEncoder

class VitSeg(nn.Module):
    def __init__(self, image_size, patch_size, hidden_size, num_layers, num_heads, mlp_dim, 
                 num_classes, feedforward_size, in_channels=3, dropout=0.1):
        # self.mae_encoder = ...
        super().__init__()
        self.seg_encoder = SegFormerEncoder(patch_size, in_channels, hidden_size, num_layers, num_heads, feedforward_size, dropout)
        # -> how many channel is it outputing??


        # TODO: scale 4 times with scale factor 4 -> to get to 4k. Figure out the best out_channel num -> chatGPT?
        # TODO calculation to get the right img size in the end, -> find out outchanel, scalefact, kernel
        patch_num = image_size // patch_size
        self.decoder = Decoder(patch_num, num_classes, hidden_size, dropout=dropout, scale_factor=4)

    def forward(self, img):
        x = self.seg_encoder(img)
        # y = self.mae_encoder(img)
        #TODO: concat x and y -> z
        # x = z
        println("shape after seg encoder", x.shape)
        x = self.decoder(x)

        return x

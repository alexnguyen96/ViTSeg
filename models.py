import torch
from torch import nn
import torch.nn.functional as F
import math

from decoder import Decoder
from encoder import SegFormerEncoder

class VitSeg(nn.Module):
    def __init__(image_size, patch_size, hidden_size, num_layers, num_heads, mlp_dim, 
                 num_classes, in_channels=3, dropout=0.1):
        # self.mae_encoder = ...
        self.seg_encoder = SegFormerEncoder(in_channels, hidden_size, 
                 num_layers, num_heads, feedforward_size, dropout)
        # -> how many channel is it outputing??

        patch_num = image_size // patch_size
        self.decoder = Decoder(patch_num, patch_size, num_classes, hidden_size, num_layers, num_heads, feedforward_size, 
                 channels=?, dropout=dropout, scale_factor=4)


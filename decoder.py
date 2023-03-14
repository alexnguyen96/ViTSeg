import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from helper import PixelShufflePack


class Decoder(nn.Module):
    def __init__(self, patch_num, num_classes, hidden_size, dropout=0.1, scale_factor=4):
        super().__init__()
        self.patch_num = patch_num

        out_chan = [hidden_size*i for i in range(2, 5)]
        self.upsampler = nn.Sequential(
          PixelShufflePack(in_channels=hidden_size, out_channels=out_chan[0], scale_factor=scale_factor),
          PixelShufflePack(in_channels=out_chan[0], out_channels=out_chan[1], scale_factor=4),
          PixelShufflePack(in_channels=out_chan[1], out_channels=out_chan[2], scale_factor=2),
          PixelShufflePack(in_channels=out_chan[2], out_channels=num_classes, scale_factor=1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_tensor):
        # so upto here: we are done with self attention and positional embedding
        # now we need remove class token


        #TODO: print the shape first to see what it is

        x = feature_tensor
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

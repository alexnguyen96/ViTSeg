import torch
from torch import nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from segformer_aiden.segformer.backbones import mit_b0 as SegFormerEncoder #Aiden's encoder

class MAEEncoder(nn.Module):
    super().__init__()
    
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import requests
import matplotlib.pyplot as plt

from models import VitSeg
from visualizer import show_result, save_result
from helper import show_image  # for MAE 1 pass run

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

# LOADING DATA
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

img_url = './data/ade_256' 
img = Image.open(img_url)
# img = Image.open(requests.get(img_url, stream=True).raw)
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

# plt.rcParams['figure.figsize'] = [5, 5]
# show_image(torch.tensor(img))

# LOADING THE WEIGHT TO MODEL
!wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth
# runcmd('wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth')


chkpt_dir = 'mae_visualize_vit_large.pth'

mae = VitSeg(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

model = mae()
# load model
checkpoint = torch.load(chkpt_dir, map_location='cpu')
msg = model.load_state_dict(checkpoint['model'], strict=False)
print(msg)
model_mae = model
print('Model loaded.')


# MAKE RANDOM MASK REPRODUCIBLE (COMMENT OUT TO MAKE IT CHANGE)
torch.manual_seed(2)
print('MAE with pixel reconstruction:')


# RUN THE MODEL
x = torch.tensor(img)

# make it a batch-like
x = x.unsqueeze(dim=0)
x = torch.einsum('nhwc->nchw', x)

# run MAE
y = model(x.float(), mask_ratio=0.75)
y = model.unpatchify(y)
y = torch.einsum('nchw->nhwc', y).detach().cpu()

print("shape of MAE encoder output", y.shape)

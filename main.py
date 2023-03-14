import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import requests
import matplotlib.pyplot as plt

from models import VitSeg

from visualizer import show_result, save_result


device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


img = Image.open('./ADE_train_00000001.jpg')

# img = Image.open('./GF2_PMS1__L1A0000564539-MSS1.tiff')
img = transforms.CenterCrop(512)(img)  # crop to get the square
img = transforms.ToTensor()(img)  # turn it into a tensor
img = torch.FloatTensor(img)[None, :]  # make it a batch

v = VitSeg(
    image_size=512,
    patch_size=32,
    num_classes=23,
    hidden_size=1024,
    num_layers=6,
    num_heads=16,
    mlp_dim=2048,
    dropout=0.1,
    scale_factor=4,
)


# dummy_result = torch.randint(23, (1, 512, 683))

#TODO: deal with sliding window, for images that are not square. for now just crop the square out of it

preds = v(img)

preds.shape

# pred_seg = preds.argmax(dim=1)
print('prediction', preds.shape)

save_result(img_path='./ADE_train_00000001.jpg',
            result=preds[0],
            class_num=23,
            out_file="./hiiiii.jpg")



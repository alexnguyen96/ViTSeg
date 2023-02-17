import torch
from models import ViTSeg

'''
TODO:
- Run to get 1 image working first
- Use the weights from the Adapter
- Try with smaller images like ADE first
'''
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

v = ViTSeg(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
)

img = torch.randn(4, 3, 256, 256)

preds = v(img)

preds.shape
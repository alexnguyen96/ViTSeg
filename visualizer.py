import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from helper import ade_palette
import matplotlib.pyplot as plt

def show_result(prediction, img_path='./ADE_train_00000001.jpg', crop_size=510):
  img = Image.open(img_path)
  image = transforms.CenterCrop(crop_size)(img)

  pred_seg = prediction
  color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
  palette = np.array(ade_palette())
  for label, color in enumerate(palette):
      color_seg[pred_seg == label, :] = color
  color_seg = color_seg[..., ::-1]  # convert to BGR

  img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
  img = img.astype(np.uint8)

  plt.figure(figsize=(15, 10))
  plt.imshow(img)
  plt.show()

def save_result(img_path,
                result,
                class_num,
                win_name='',
                wait_time=0,
                out_file=None,
                opacity=0.5):
    """Draw `result` over `img`.

    Args:
        img (str): Path to the image
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        nothing, save the img to outfile
    """

    assert (out_file), 'Please specify where you want the image to be saved'

    img = Image.open(img_path)

    size = result.shape[0]

    img = transforms.CenterCrop(size)(img)  # crop to get the square
    img = np.asarray(img)
    img = img.copy()
    seg = result  # change this number to view the result of the other inputs

    # Get random state before set seed,
    # and restore random state later.
    # It will prevent loss of randomness, as the palette
    # may be different in each iteration if not specified.
    # See: https://github.com/open-mmlab/mmdetection/issues/5844
    state = np.random.get_state()
    np.random.seed(42)
    
    # random palette
    palette = np.random.randint(
        0, 255, size=(class_num, 3))
    np.random.set_state(state)

    palette = np.array(palette)
    assert palette.shape[0] == class_num
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    data = Image.fromarray(img)
    data.save(out_file)


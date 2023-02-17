import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def show_result(img,
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
        img (Tensor): Only if not `show` or `out_file`
    """

    assert (out_file), 'Please specify where you want the image to be saved'

    img = Image.open(img)
    # img = transforms.CenterCrop(512)(img)  # crop to get the square
    img = np.asarray(img)
    img = img.copy()
    seg = result[0]  # change this number to view the result of the other inputs

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
    # if out_file specified, do not show image in window
    # if out_file is not None:
    #     show = False

    data = Image.fromarray(img)
    data.save(out_file)
    # if show:
    #     mmcv.imshow(img, win_name, wait_time)
    # if out_file is not None:
    #     mmcv.imwrite(img, out_file)

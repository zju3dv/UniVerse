"""Degradations used to synthesise inconsistent training frames.

Every degraded frame gets a random photometric jitter (brightness, contrast, saturation, hue and
sharpness, applied in a random order) followed by exactly one of: motion blur, Gaussian blur,
Gaussian noise, occlusion by random object masks (4x more likely) or nothing.
"""
import numbers
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import GaussianBlur
from torchvision.transforms import functional as transF

MASK_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


# Adapted from torchvision.transforms.ColorJitter (BSD-3-Clause, https://github.com/pytorch/vision).
class AugJitter(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation, hue and sharpness of an image.

    All five ops are applied, in a random order. The parameters of the last call (including the
    op order ``idx``) are kept so that ``apply_deterministic`` can replay exactly the same jitter.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
        hue (float or tuple of float (min, max)): How much to jitter hue.
        sharpness (float or tuple of float (min, max)): How much to jitter sharpness.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, sharpness=0):
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.sharpness = self._check_input(sharpness, "sharpness")

        self._last_applied_params = None

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
        sharpness: Optional[List[float]]
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image."""
        fn_idx = torch.randperm(5)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))
        sh = None if sharpness is None else float(torch.empty(1).uniform_(sharpness[0], sharpness[1]))

        return fn_idx, b, c, s, h, sh

    def apply_deterministic(self, img, params):
        """Replay a jitter with the given parameters, in the stored op order ``params['idx']``.

        Works on a single (3, H, W) image or a batch of frames (T, 3, H, W).
        """
        for fn_id in params["idx"]:
            if fn_id == 0 and params.get("brightness") is not None:
                img = transF.adjust_brightness(img, params["brightness"])
            elif fn_id == 1 and params.get("contrast") is not None:
                img = transF.adjust_contrast(img, params["contrast"])
            elif fn_id == 2 and params.get("saturation") is not None:
                img = transF.adjust_saturation(img, params["saturation"])
            elif fn_id == 3 and params.get("hue") is not None:
                img = transF.adjust_hue(img, params["hue"])
            elif fn_id == 4 and params.get("sharpness") is not None:
                img = transF.adjust_sharpness(img, params["sharpness"])
        return img

    def forward(self, img):
        """Apply the color jittering to the input image."""
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, sharpness_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue, self.sharpness
        )

        self._last_applied_params = {
            "idx": fn_idx,
            "brightness": brightness_factor,
            "contrast": contrast_factor,
            "saturation": saturation_factor,
            "hue": hue_factor,
            "sharpness": sharpness_factor
        }

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = transF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = transF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = transF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = transF.adjust_hue(img, hue_factor)
            elif fn_id == 4 and sharpness_factor is not None:
                img = transF.adjust_sharpness(img, sharpness_factor)

        return img

    def get_applied_params(self):
        """Return the parameters used in the last transform."""
        return self._last_applied_params

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"hue={self.hue}, "
            f"sharpness={self.sharpness})"
        )
        return s


# Training jitter: brightness U[0.3, 2], contrast U[0.7, 2], saturation U[0.7, 1.3],
# hue U[-0.15, 0.15], sharpness U[0.7, 4].
jitter = AugJitter(brightness=[0.3, 2], contrast=[0.7, 2], hue=0.15, saturation=0.3, sharpness=[0.7, 4])


def _kernel_size_range(height, width):
    """Range of the (odd) blur kernel sizes: 3-19 px at 320x512 and 3-41 px at 576x1024. The lower
    bound is forced to be odd (and the range non-empty) so that other resolutions work as well; the
    two training resolutions are unchanged."""
    low = int(height / 320) * 3 | 1
    return low, max(int(width / 512) * 21, low + 2)


def motion_blur(input_tensor):
    """Linear motion blur with a random angle; the kernel length scales with the resolution
    (3-19 px at 320x512, 3-41 px at 576x1024). input_tensor: (C, H, W)."""
    input_tensor = input_tensor[None]
    kernel_size = random.randrange(*_kernel_size_range(*input_tensor.shape[-2:]), 2)

    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    center = kernel_size // 2

    for i in range(kernel_size):
        kernel[i, center] = 1.0
    angle = random.uniform(0., 180.)
    theta = torch.tensor(angle / 180.0 * torch.pi)
    rotation_matrix = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0]
    ])

    grid = F.affine_grid(rotation_matrix.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).size(), align_corners=False)
    kernel = F.grid_sample(kernel.unsqueeze(0).unsqueeze(0), grid, align_corners=False).squeeze()

    kernel /= kernel.sum()

    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(input_tensor.size(1), 1, 1, 1)

    padding = kernel_size // 2
    blurred_tensor = F.conv2d(input_tensor, kernel, padding=padding, groups=input_tensor.size(1))

    return blurred_tensor.squeeze()


def gaussian_blur(img: torch.Tensor):
    """Gaussian blur, sigma U[0.1, 2]; same kernel-size range as ``motion_blur``."""
    gaussian = GaussianBlur(kernel_size=random.randrange(*_kernel_size_range(*img.shape[-2:]), 2),
                            sigma=random.uniform(0.1, 2.0))
    return gaussian(img)


def gaussian_noise(img: torch.Tensor):
    """Additive Gaussian noise, std U[10, 50] on the 0-255 scale (quantised to uint8)."""
    if img.max() <= 1.0:
        img = (img * 255)
    std = random.uniform(10, 50)
    img = (img + torch.randn_like(img) * std).clip(0, 255.).to(torch.uint8)
    return img / 255.


def list_mask_files(mask_dirs):
    """Occluder mask pool: one sorted list of mask images per directory."""
    if isinstance(mask_dirs, str):
        mask_dirs = [mask_dirs]
    if not mask_dirs:
        raise ValueError("At least one occluder mask directory is required.")
    pool = []
    for mask_dir in mask_dirs:
        files = sorted(os.path.join(mask_dir, name) for name in os.listdir(mask_dir)
                       if name.lower().endswith(MASK_EXTS))
        if not files:
            raise FileNotFoundError(f"No mask images {MASK_EXTS} found in {mask_dir}")
        pool.append(files)
    return pool


def load_mask(path):
    """Occluder mask as an (H, W) uint8 tensor; values > 0 are the object.

    Single-channel ('L') and palette ('P', e.g. VOC SegmentationClass) images are used as stored.
    Images whose alpha channel outlines the object (e.g. RGBA cut-outs on a transparent background)
    use the alpha channel; all others are converted to grayscale.
    """
    with Image.open(path) as mask:
        if mask.mode in ('L', 'P'):
            return torch.from_numpy(np.array(mask))
        if 'A' in mask.getbands():
            alpha = np.array(mask.getchannel('A'))
            if alpha.min() == 0 and alpha.max() > 0:
                return torch.from_numpy(alpha)
        return torch.from_numpy(np.array(mask.convert('L')))


def occlusions(image, mask_pool):
    """Black out 1-2 random object masks.

    Each mask is drawn from ``mask_pool`` (a directory uniformly, then a file), resized with nearest
    interpolation to a random box of [H/9, H] x [W/9, W] pixels (transposed to match the box
    orientation) and pasted at a random position. Returns the occluded image and the (3, H, W)
    bool occlusion mask.
    """
    def random_mask(image_size):
        mask_now = load_mask(random.choice(random.choice(mask_pool)))
        mask = torch.zeros(image_size)

        h = random.randint(image_size[-2] // 9, image_size[-2])
        w = random.randint(image_size[-1] // 9, image_size[-1])

        if (mask_now.shape[0] > mask_now.shape[1]) != (h > w):
            mask_now = mask_now.transpose(-1, -2)
        mask_now = F.interpolate(mask_now[None][None], size=(h, w)).squeeze()

        x1 = random.randint(0, image_size[0] - h)
        y1 = random.randint(0, image_size[1] - w)
        mask[x1:x1 + h, y1:y1 + w] = mask_now
        return mask > 0

    mask_nums = random.randint(1, 2)
    mask = torch.zeros((image.shape[-2], image.shape[-1]), dtype=bool)
    for _ in range(mask_nums):
        mask = mask | random_mask((image.shape[-2], image.shape[-1]))
    image_with_mask = image.clone()
    image_with_mask[:, mask] = 0
    return image_with_mask, mask[None].repeat(3, 1, 1)


def identity(img):
    return img


def random_deg(img: torch.Tensor, mask_pool):
    """Degrade one (3, H, W) frame in [0, 1]: jitter, then one of motion blur / Gaussian blur /
    Gaussian noise / occlusion (x4) / identity.

    Returns the degraded frame, its occlusion mask (all zeros unless occluded) and the jitter
    parameters (for ``jitter.apply_deterministic``).
    """
    deg_list = [motion_blur, gaussian_blur, gaussian_noise] + [occlusions] * 4 + [identity]
    img = jitter(img.cpu())
    deg = random.choice(deg_list)
    mask = torch.zeros_like(img)
    if deg is occlusions:
        img, mask = deg(img, mask_pool)
    else:
        img = deg(img)
    return img.clip(0, 1.), mask, jitter.get_applied_params()

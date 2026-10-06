"""Turn a clean multi-view scene into an inconsistent UniVerse test scene.

Builds scenes like the synthetic LLFF benchmark of the paper (Sec. 4, "Dataset"; 20-50 views per scene).
Every degraded view gets the photometric jitter used in training (brightness, contrast, saturation, hue and
sharpness, applied in a random order) and one of the following: a PASCAL VOC2007 object pasted over most of
the frame (--occlusion_prob, 2/3 by default), motion blur, Gaussian blur, Gaussian noise, or nothing. Two
kinds of views stay clean:

  hold-out  every --llffhold-th view by file name (LLFF protocol), for novel-view evaluation;
  anchor    the first of the other views in the ThreadPose order of inference.py. This is the style image
            of the restoration, so the restored scene keeps the colours of the clean scene.

Input: <scene_dir>/images and <scene_dir>/sparse/0 (COLMAP) or <scene_dir>/transforms.json. Output:

  <out_dir>/images/        all views at --height x --width (resized to cover, center-cropped, names kept)
  <out_dir>/images_train/  copies of the views to restore, i.e. all but the hold-out views (--llffhold > 0)
  <out_dir>/gt/            the clean views, same crop
  <out_dir>/masks/         <image stem>.png for every view: 255 = pasted occluder (inpaint), 0 = keep
  <out_dir>/sparse/, <out_dir>/transforms.json   copied unchanged
  <out_dir>/degradation_log.json                 split, ops and jitter parameters of every view

inference.py only reads the camera poses of the copied model, to order and space the views. Its intrinsics
refer to the original resolution unless the scene already is --height x --width, and it still lists the views
dropped by --max_images. For a reconstruction, re-run COLMAP, as the paper does on the restored images.

VOC2007: extract VOCtrainval_06-Nov-2007.tar (http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) and pass its
VOCdevkit/VOC2007 folder. Objects are taken from SegmentationObject/*.png, with their pixels from JPEGImages/.

Example:
    python scripts/synthesize_test_scene.py --scene_dir nerf_llff_data/fern --out_dir test_scenes/fern \
        --voc_root VOCdevkit/VOC2007
    python inference.py --image_dir test_scenes/fern/images_train --out_dir output/fern
"""

import argparse
import json
import math
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import (list_images, read_colmap_images, read_transforms_json, resize_crop, save_image,  # noqa: E402
                       str2bool, thread_pose)
from lvdm.data.degradations import (gaussian_blur, gaussian_noise, identity, jitter, load_mask,  # noqa: E402
                                    motion_blur)

JITTER_OPS = ('brightness', 'contrast', 'saturation', 'hue', 'sharpness')  # AugJitter op ids 0-4
OTHER_OPS = (motion_blur, gaussian_blur, gaussian_noise, identity)  # equally likely when not occluded
OUTPUT_DIRS = ('images', 'images_train', 'gt', 'masks')


def select_views(names, max_images):
    """Keep every k-th view, with the smallest k that leaves at most ``max_images`` (0 keeps all)."""
    if max_images <= 0 or len(names) <= max_images:
        return names
    return names[::math.ceil(len(names) / max_images)]


def find_poses(scene_dir):
    """(COLMAP model dir, transforms.json) of a scene, looked up like inference.prepare_scene does."""
    colmap = os.path.join(scene_dir, 'sparse', '0')
    transforms = os.path.join(scene_dir, 'transforms.json')
    if os.path.isdir(colmap):
        return colmap, None
    if os.path.exists(transforms):
        return None, transforms
    return None, None


def trajectory_order(names, image_dir, colmap=None, transforms=None):
    """ThreadPose order of the given views of ``image_dir``, as inference.order_views computes it for a
    folder holding exactly these images. Views without a pose are left out; without poses the file-name
    order is kept."""
    if colmap is not None:
        poses = sorted(read_colmap_images(colmap), key=lambda p: p[0])
    elif transforms is not None:
        poses = [(os.path.basename(path), pose) for path, pose in read_transforms_json(transforms, image_dir)]
    else:
        return sorted(names)
    names = set(names)
    poses = [(name, pose) for name, pose in poses if name in names]
    if not poses:
        return []
    order, _ = thread_pose(torch.tensor(np.stack([pose for _, pose in poses])))
    return [poses[i][0] for i in order.tolist()]


class VOCObjects:
    """Occluders from PASCAL VOC: SegmentationObject/<id>.png masks with the pixels of JPEGImages/<id>.jpg.

    As in the research code, only images with the frame's orientation (landscape or portrait) are used.
    """

    def __init__(self, voc_root, portrait=False):
        self.mask_dir = os.path.join(voc_root, 'SegmentationObject')
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        for folder in (self.mask_dir, self.image_dir):
            if not os.path.isdir(folder):
                raise FileNotFoundError(f'{folder} not found; --voc_root must be the VOCdevkit/VOC2007 folder')
        self.ids = []
        for name in sorted(os.listdir(self.mask_dir)):
            voc_id, ext = os.path.splitext(name)
            if ext.lower() != '.png' or not os.path.exists(os.path.join(self.image_dir, voc_id + '.jpg')):
                continue
            with Image.open(os.path.join(self.mask_dir, name)) as mask:
                w, h = mask.size
            if (h > w) == portrait:
                self.ids.append(voc_id)
        if not self.ids:
            raise FileNotFoundError(f'No usable object masks in {self.mask_dir}')

    def paste(self, image):
        """Paste a random object over most of a (3, H, W) image (``occlusions_real`` of the research code).

        The object and its mask are resized (nearest) to a random box of [H / 1.01, H] rows with the frame's
        aspect ratio, placed at a random position and pasted where the mask is > 0 (the objects plus the VOC
        'void' outline). Returns the image, the (H, W) bool occluder mask and the log entry.
        """
        height, width = image.shape[-2:]
        voc_id = random.choice(self.ids)
        mask = load_mask(os.path.join(self.mask_dir, voc_id + '.png'))
        with Image.open(os.path.join(self.image_dir, voc_id + '.jpg')) as obj:
            obj = torch.from_numpy(np.array(obj.convert('RGB'))).permute(2, 0, 1).float() / 255.
        rows = random.randint(int(height // 1.01), height)
        cols = min(width, int(rows * width / height))
        mask = F.interpolate(mask[None, None].float(), size=(rows, cols))[0, 0] > 0
        obj = F.interpolate(obj[None], size=(rows, cols))[0]
        y0, x0 = random.randint(0, height - rows), random.randint(0, width - cols)

        image = image.clone()
        image[:, y0:y0 + rows, x0:x0 + cols][:, mask] = obj[:, mask]
        occluder = torch.zeros(height, width, dtype=torch.bool)
        occluder[y0:y0 + rows, x0:x0 + cols] = mask
        return image, occluder, dict(voc_id=voc_id, box_yxhw=[y0, x0, rows, cols])


def degrade_view(image, voc, occlusion_prob):
    """Degrade a clean (3, H, W) view in [0, 1] (``random_deg_real`` of the research code).

    With probability ``occlusion_prob`` a VOC object is pasted and the whole image is then jittered;
    otherwise the image is jittered, then motion-blurred, Gaussian-blurred, noised or left as is.
    Returns the degraded view, the (H, W) bool occluder mask and the log entry.
    """
    mask = torch.zeros(image.shape[-2:], dtype=torch.bool)
    if voc is not None and random.random() < occlusion_prob:
        image, mask, occluder = voc.paste(image)
        image = jitter(image)
        entry = dict(ops=['occlusion', 'jitter'], occluder=occluder)
    else:
        op = random.choice(OTHER_OPS)
        image = op(jitter(image))
        entry = dict(ops=['jitter'] + ([op.__name__] if op is not identity else []))
    params = jitter.get_applied_params()
    entry['jitter'] = dict(order=[JITTER_OPS[i] for i in params['idx'].tolist()], **{k: params[k] for k in JITTER_OPS})
    return image.clip(0, 1.), mask, entry


def seed_view(seed, scene, name):
    """Seed Python's and torch's RNGs from (seed, scene folder name, image name), so a view's degradation does
    not depend on the other views and scenes with the same file names are degraded differently. Outputs are
    reproducible on the same machine and thread count."""
    random.seed(f'{seed}/{scene}/{name}')
    torch.manual_seed(random.getrandbits(32))


def load_clean_view(path, height, width, resize_mode):
    """(3, height, width) float image in [0, 1], resized to cover and center-cropped like inference.py.
    EXIF orientation is ignored, as in the research code."""
    with Image.open(path) as image:
        image = torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1)[None] / 255.
    return resize_crop(image, height, width, resize_mode)[0]


def save_mask(mask, path):
    """Save an (H, W) bool mask as a single-channel 0/255 PNG."""
    Image.fromarray(mask.numpy().astype(np.uint8) * 255).save(path)


def get_parser():
    parser = argparse.ArgumentParser(description='Synthesize an inconsistent UniVerse test scene from a clean scene.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='Clean scene with images/ and sparse/0 (COLMAP) or transforms.json.')
    parser.add_argument('--out_dir', type=str, required=True, help='Output scene directory.')
    parser.add_argument('--voc_root', type=str, default=None,
                        help='VOCdevkit/VOC2007 folder (SegmentationObject/ + JPEGImages/). Without it occlusions '
                             'are disabled.')
    parser.add_argument('--height', type=int, default=576, help='Output image height.')
    parser.add_argument('--width', type=int, default=1024, help='Output image width.')
    parser.add_argument('--resize_mode', type=str, default='nearest', choices=['nearest', 'bilinear', 'bicubic'],
                        help='Interpolation used to resize the views (bilinear and bicubic are antialiased).')
    parser.add_argument('--max_images', type=int, default=50,
                        help='Keep every k-th view (by file name) so that at most this many remain; 0 keeps all.')
    parser.add_argument('--llffhold', type=int, default=8,
                        help='Every llffhold-th view (by file name) is a clean hold-out view; 0 disables.')
    parser.add_argument('--keep_anchor', type=str2bool, default=True,
                        help='Keep the first view of the restoration order (the style image) clean.')
    parser.add_argument('--occlusion_prob', type=float, default=2 / 3,
                        help='Probability that a degraded view gets a pasted VOC object instead of motion blur, '
                             'Gaussian blur, Gaussian noise or nothing (equally likely). (default: %(default).3f)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed. Each degraded view is seeded from it, the scene folder name and the image '
                             'file name; outputs are reproducible on the same machine and thread count.')
    return parser


def main():
    args = get_parser().parse_args()
    scene_dir = os.path.normpath(args.scene_dir)
    scene = os.path.basename(os.path.abspath(scene_dir))  # part of the per-view seeds
    image_dir = os.path.join(scene_dir, 'images')
    out_dir = os.path.normpath(args.out_dir)
    if not os.path.isdir(image_dir):
        sys.exit(f'Error: {image_dir} not found')
    if os.path.isdir(out_dir) and os.path.samefile(out_dir, scene_dir):
        sys.exit('Error: --out_dir must differ from --scene_dir')
    for sub in OUTPUT_DIRS:
        if os.path.isdir(os.path.join(out_dir, sub)) and os.listdir(os.path.join(out_dir, sub)):
            sys.exit(f'Error: {os.path.join(out_dir, sub)} is not empty; choose a new --out_dir')
    if args.height < 320 or args.width < 512:
        sys.exit('Error: the blur kernels need --height >= 320 and --width >= 512')
    if not 0 <= args.occlusion_prob <= 1 or args.llffhold < 0:
        sys.exit('Error: --occlusion_prob must be in [0, 1] and --llffhold >= 0')

    names = select_views(list_images(image_dir), args.max_images)
    if not names:
        sys.exit(f'Error: no images in {image_dir}')
    mask_names = [os.path.splitext(name)[0] + '.png' for name in names]
    if len(set(mask_names)) < len(names):
        sys.exit('Error: several images share a file stem, so their masks would collide')
    holdout = set(names[::args.llffhold]) if args.llffhold > 0 else set()
    train = [name for name in names if name not in holdout]

    # The anchor is the first view in the order inference.py restores the training views in.
    colmap, transforms = find_poses(scene_dir)
    if colmap is None and transforms is None:
        print(f'Warning: no sparse/0 or transforms.json in {scene_dir}; views are ordered by file name')
    order = trajectory_order(train, image_dir, colmap, transforms)
    unposed = [name for name in train if name not in order]
    if unposed:
        print(f'Warning: {len(unposed)} views have no pose and will not be restored, e.g. {unposed[:3]}')
    if len(order) < 2:
        sys.exit(f'Error: UniVerse needs at least 2 views to restore, got {len(order)}: {len(names)} selected, '
                 f'{len(holdout)} hold-out, {len(unposed)} without a pose')
    anchor = order[0] if args.keep_anchor else None

    voc = None
    if args.voc_root is None:
        print('Warning: no --voc_root given, occlusions are disabled')
    elif args.occlusion_prob > 0:
        try:
            voc = VOCObjects(args.voc_root, portrait=args.height > args.width)
        except FileNotFoundError as e:
            sys.exit(f'Error: {e}')
        print(f'{len(voc.ids)} VOC object images with the frame orientation in {voc.mask_dir}')
    print(f'{len(names)} views: {len(holdout)} hold-out, anchor {anchor}, {len(train) - (anchor is not None)} degraded')

    for sub in ('images', 'gt', 'masks'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    entries, num_occluded = {}, 0
    for i, (name, mask_name) in enumerate(zip(names, mask_names)):
        clean = load_clean_view(os.path.join(image_dir, name), args.height, args.width, args.resize_mode)
        gt_path, image_path = os.path.join(out_dir, 'gt', name), os.path.join(out_dir, 'images', name)
        save_image(clean.permute(1, 2, 0), gt_path)
        if name in holdout or name == anchor:
            mask, entry = torch.zeros(clean.shape[-2:], dtype=torch.bool), dict(ops=[])
            entry['split'] = 'holdout' if name in holdout else 'anchor'
            shutil.copyfile(gt_path, image_path)
        else:
            seed_view(args.seed, scene, name)
            image, mask, entry = degrade_view(clean, voc, args.occlusion_prob)
            assert image.shape == clean.shape and mask.shape == clean.shape[-2:], \
                f'{name}: degraded view {tuple(image.shape)} and mask {tuple(mask.shape)} vs gt {tuple(clean.shape)}'
            entry['split'] = 'degraded'
            save_image(image.permute(1, 2, 0), image_path)
        save_mask(mask, os.path.join(out_dir, 'masks', mask_name))
        entry['occluded_fraction'] = round(mask.sum().item() / mask.numel(), 6)
        num_occluded += bool(mask.any())
        entries[name] = {key: entry[key] for key in ('split', 'ops', 'jitter', 'occluder', 'occluded_fraction')
                         if key in entry}
        print(f'[{i + 1}/{len(names)}] {name}: {entry["split"]} {" + ".join(entry["ops"])}'
              + (f' ({entry["occluded_fraction"]:.1%} occluded)' if mask.any() else ''))

    # inference.py restores the views of an image folder: without the hold-out views when there are any.
    restore_dir = os.path.join(out_dir, 'images')
    if holdout:
        restore_dir = os.path.join(out_dir, 'images_train')
        os.makedirs(restore_dir, exist_ok=True)
        for name in train:
            shutil.copyfile(os.path.join(out_dir, 'images', name), os.path.join(restore_dir, name))
    if os.path.isdir(os.path.join(scene_dir, 'sparse')):
        shutil.copytree(os.path.join(scene_dir, 'sparse'), os.path.join(out_dir, 'sparse'), dirs_exist_ok=True)
    if os.path.exists(os.path.join(scene_dir, 'transforms.json')):
        shutil.copyfile(os.path.join(scene_dir, 'transforms.json'), os.path.join(out_dir, 'transforms.json'))

    log = dict(args=vars(args), scene=scene, holdout=sorted(holdout), anchor=anchor,
               restore_image_dir=os.path.basename(restore_dir), restoration_order=order, views=entries)
    with open(os.path.join(out_dir, 'degradation_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print(f'Done: {len(names)} views in {out_dir} ({num_occluded} with occluders). Restore them with:\n'
          f'  python inference.py --image_dir {restore_dir} --out_dir <output dir>')


if __name__ == '__main__':
    main()

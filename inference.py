"""UniVerse inference: restore inconsistent multi-view images into consistent ones.

UniVerse: Unleashing the Scene Prior of Video Diffusion Models for Robust Radiance
Field Reconstruction (ICCV 2025) -- https://arxiv.org/abs/2510.01669

This file is the complete restoration pipeline; it is a cleaned-up, self-contained
version of the research notebook ``inference.ipynb``. Given K inconsistent images of a
static scene (varying exposure / white balance / filters, blur, noise, transient
occluders), optional inpainting masks and rough camera poses, it

  1. sorts the images along an implicit camera trajectory (ThreadPose, Supp. Alg. 2);
  2. turns every batch of N images into a 25-frame *initial video* by inserting zero
     frames between neighbouring images, proportionally to their pose distance;
  3. restores the initial video with the mask- and style-conditioned video diffusion
     model: VAE latents + inpainting mask + style mask + noise go through a 10-channel
     U-Net, and the CLIP embeddings of the N inputs go through the Multi-input Query
     Transformer;
  4. keeps the N restored frames and continues with the next batch, whose first frame
     and style reference is the last restored image (Supp. Alg. 1).

The restored images keep the input file names (at the model resolution, center-cropped)
and can be fed to any 3D reconstruction method; the paper re-runs COLMAP on them and
trains Zip-NeRF with GLO.

load_model_checkpoint, get_latent_z, image_guided_synthesis and the UniVerse model wrapper
are adapted from ViewCrafter (https://github.com/Drexubery/ViewCrafter, Apache License 2.0)
and modified for UniVerse (inpainting / style masks, Multi-input Query Transformer).

Example:
    python inference.py --image_dir data/demo/images --out_dir output/demo
"""

import argparse
import json
import os
import struct
from collections import Counter, OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from scipy.spatial.transform import Rotation
from torchvision.transforms import CenterCrop

from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSamplerMultiCond
from lvdm.utils import instantiate_from_config

# The video model generates 25 frames. Keep it fixed: the U-Net splits the cross-attention
# context per frame when it has exactly 77 + 16 * T tokens, which would wrongly trigger for T = 16.
NUM_FRAMES = 25
ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
DEFAULT_PROMPT = 'A consistent scene captured using a continuous camera trajectory'
MODEL_PRESETS = {
    '512': dict(config='configs/inference_512.yaml', ckpt='checkpoints/universe_512.ckpt'),
    '1024': dict(config='configs/inference_1024.yaml', ckpt='checkpoints/universe_1024.ckpt'),
}


# ----------------------------------------------------------------------------------------
# Camera poses and ThreadPose ordering
# ----------------------------------------------------------------------------------------

def read_colmap_images(path):
    """Read image poses from a COLMAP ``images.txt`` / ``images.bin`` (or a sparse model dir).

    Returns a list of (image name, 4x4 world-to-camera matrix) in file order.
    """
    if os.path.isdir(path):
        for fname in ('images.txt', 'images.bin'):
            if os.path.exists(os.path.join(path, fname)):
                path = os.path.join(path, fname)
                break
        else:
            raise FileNotFoundError(f'No images.txt / images.bin in {path}')

    entries = []
    if path.endswith('.bin'):
        with open(path, 'rb') as f:
            num_images = struct.unpack('<Q', f.read(8))[0]
            for _ in range(num_images):
                props = struct.unpack('<idddddddi', f.read(64))
                name = b''
                char = f.read(1)
                while char != b'\x00':
                    name += char
                    char = f.read(1)
                num_points2d = struct.unpack('<Q', f.read(8))[0]
                f.read(24 * num_points2d)  # (x, y, point3D_id) per observation
                entries.append((name.decode('utf-8'), props[1:5], props[5:8]))
    else:
        # Two lines per image: "IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", then the 2D points
        # (possibly an empty line), as in COLMAP's own read_images_text.
        with open(path, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                parts = line.split()
                if not parts or parts[0].startswith('#'):
                    continue
                if len(parts) >= 10:
                    entries.append((' '.join(parts[9:]), tuple(map(float, parts[1:5])),
                                    tuple(map(float, parts[5:8]))))
                f.readline()  # 2D points

    poses = []
    for name, (qw, qx, qy, qz), tvec in entries:
        w2c = np.eye(4)
        w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, 3] = tvec
        poses.append((name, w2c))
    return poses


def _with_extension(path):
    """Blender-style ``transforms.json`` files omit the image extension."""
    if os.path.splitext(path)[1] or os.path.exists(path):
        return path
    for ext in ('.png', '.jpg', '.jpeg', '.JPG', '.PNG'):
        if os.path.exists(path + ext):
            return path + ext
    return path


def read_transforms_json(path, image_dir=None):
    """Read (image path, 4x4 camera-to-world matrix) pairs from a NeRF-style ``transforms.json``.

    Images are looked up by file name in ``image_dir`` if given (only the images of that folder
    are restored), otherwise relative to the json file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    root = os.path.dirname(os.path.abspath(path))
    poses = []
    for frame in meta['frames']:
        if image_dir is not None:
            file_path = os.path.join(image_dir, os.path.basename(frame['file_path']))
        else:
            file_path = os.path.normpath(os.path.join(root, frame['file_path']))
        poses.append((_with_extension(file_path), np.array(frame['transform_matrix'], dtype=np.float64)))
    return poses


def rotation_distance(R1, R2):
    """Geodesic distance (radians) between two rotation matrices."""
    trace = torch.trace(R1 @ R2.transpose(0, 1))
    cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    return torch.acos(cos_theta)


def thread_pose(poses, rotation_weight=0.5, translation_weight=0.5):
    """ThreadPose: order poses along an implicit camera trajectory (Supp. Alg. 2).

    A doubly linked list is grown greedily from pose 0: at each step the unvisited pose that
    is closest to the head or to the tail of the list is attached to that end. The pose
    distance mixes the rotation geodesic and the translation distance, each normalised by
    its maximum over all pairs. As in the research code, the translations are the pose
    matrices' own (world-to-camera for COLMAP, camera centres for transforms.json).

    Args:
        poses: (K, 4, 4) or (K, 3, 4) tensor.
    Returns:
        order: (K,) long tensor, pose indices in trajectory order.
        distances: (K-1,) tensor, distance between consecutive poses of the ordering.
    """
    rotations, translations = poses[:, :3, :3], poses[:, :3, 3]
    n = poses.shape[0]
    rotation_matrix = torch.zeros((n, n))
    translation_matrix = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rotation_matrix[i, j] = rotation_distance(rotations[i], rotations[j])
            translation_matrix[i, j] = torch.norm(translations[i] - translations[j])

    max_rotation, max_translation = rotation_matrix.max(), translation_matrix.max()
    if max_rotation > 0:
        rotation_matrix = rotation_matrix / max_rotation
    if max_translation > 0:
        translation_matrix = translation_matrix / max_translation
    distance_matrix = rotation_weight * rotation_matrix + translation_weight * translation_matrix
    pair_distance = distance_matrix.clone()

    visited = torch.zeros(n, dtype=torch.bool)
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        head, tail = order[0], order[-1]
        distance_matrix[head, visited] = float('inf')
        distance_matrix[tail, visited] = float('inf')
        if distance_matrix[head].min() < distance_matrix[tail].min():
            nearest = torch.argmin(distance_matrix[head]).item()
            order.insert(0, nearest)
        else:
            nearest = torch.argmin(distance_matrix[tail]).item()
            order.append(nearest)
        visited[nearest] = True

    distances = [pair_distance[order[i - 1], order[i]] for i in range(1, len(order))]
    return torch.tensor(order), torch.tensor(distances)


def list_images(image_dir):
    return sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS))


def order_views(image_dir, colmap=None, transforms=None):
    """Order the views of a scene.

    Returns:
        views: list of (image path, output name) in trajectory order.
        distances: (K-1,) tensor of distances between consecutive views.
    """
    if colmap is not None:
        poses = sorted(read_colmap_images(colmap), key=lambda p: p[0])  # name order, then ThreadPose
        views = [(os.path.join(image_dir, name), name) for name, _ in poses]
    elif transforms is not None:
        poses = read_transforms_json(transforms, image_dir)
        views = [(path, os.path.basename(path)) for path, _ in poses]
        counts = Counter(name for _, name in views)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f'{transforms} lists several images named {duplicates[:3]}; restored images are '
                             f'saved by file name, so the names must be unique.')
    else:
        # No poses: keep the file-name order and space the views uniformly.
        names = list_images(image_dir)
        return [(os.path.join(image_dir, n), n) for n in names], torch.ones(max(len(names) - 1, 0))

    exists = [os.path.exists(path) for path, _ in views]
    if not all(exists):
        missing = [name for (_, name), e in zip(views, exists) if not e]
        print(f'Warning: {len(missing)} posed images not found and skipped, e.g. {missing[:3]}')
        poses = [p for p, e in zip(poses, exists) if e]
        views = [v for v, e in zip(views, exists) if e]
    if image_dir is not None:
        unposed = sorted(set(list_images(image_dir)) - {os.path.basename(name) for _, name in views})
        if unposed:
            print(f'Warning: {len(unposed)} images have no pose and are not restored, e.g. {unposed[:3]}')
    if not views:
        raise FileNotFoundError(f'No posed images found in {image_dir}')

    order, distances = thread_pose(torch.tensor(np.stack([pose for _, pose in poses])))
    return [views[i] for i in order.tolist()], distances


def prepare_scene(image_dir, mask_dir=None, colmap=None, transforms=None, no_poses=False):
    """Resolve the inputs of a scene laid out as ``<scene>/images``, ``<scene>/masks`` (optional)
    and ``<scene>/sparse/0`` (COLMAP) or ``<scene>/transforms.json``.

    Returns (views in trajectory order, consecutive pose distances, mask dir or None).
    """
    image_dir = os.path.normpath(image_dir)
    scene_dir = os.path.dirname(image_dir)
    if mask_dir is not None and not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'Mask directory not found: {mask_dir}')
    if mask_dir is None and os.path.isdir(os.path.join(scene_dir, 'masks')):
        mask_dir = os.path.join(scene_dir, 'masks')
    if no_poses:
        colmap = transforms = None
    elif colmap is None and transforms is None:
        if os.path.isdir(os.path.join(scene_dir, 'sparse', '0')):
            colmap = os.path.join(scene_dir, 'sparse', '0')
        elif os.path.exists(os.path.join(scene_dir, 'transforms.json')):
            transforms = os.path.join(scene_dir, 'transforms.json')
        else:
            print('Warning: no camera poses found (<scene>/sparse/0 or <scene>/transforms.json); the images are '
                  'taken in file-name order and spaced uniformly. Pass --no_poses to silence this warning.')
    views, distances = order_views(image_dir, colmap=colmap, transforms=transforms)
    source = 'COLMAP poses' if colmap else 'transforms.json' if transforms else 'file names (no poses)'
    print(f'{len(views)} views ordered by {source}; masks: {mask_dir or "none"}')
    if mask_dir is not None:
        no_mask = [name for _, name in views if find_mask(mask_dir, name) is None]
        if no_mask:
            print(f'Warning: no mask for {len(no_mask)} images (nothing is inpainted there), e.g. {no_mask[:3]}')
    return views, distances, mask_dir


# ----------------------------------------------------------------------------------------
# Initial video construction
# ----------------------------------------------------------------------------------------

def allocate_weights_to_integers(weights, total):
    """Split ``total`` into integers proportional to ``weights``; the remainder goes to the
    entries with the largest fractional parts."""
    if weights.sum() <= 0:
        weights = torch.ones_like(weights)
    scaled_values = weights / weights.sum() * total
    integer_parts = torch.floor(scaled_values).to(torch.int32)
    remainder = total - integer_parts.sum()
    fractional_parts = scaled_values - integer_parts
    sorted_indices = torch.argsort(fractional_parts, descending=True)
    for i in range(remainder):
        integer_parts[sorted_indices[i]] += 1
    return integer_parts


def frame_positions(distances, num_frames=NUM_FRAMES):
    """Frame index of each of the N = len(distances) + 1 views inside the initial video.

    The num_frames - N zero frames are spread over the N - 1 gaps proportionally to the pose
    distance of each gap; the first view is frame 0 and the last view is frame num_frames - 1.
    """
    gaps = allocate_weights_to_integers(distances, num_frames - (len(distances) + 1)) + 1
    positions = [0]
    for gap in gaps:
        positions.append(positions[-1] + int(gap))
    return positions


def images_per_iteration(num_images, num_frames=NUM_FRAMES):
    """Paper rule: N = floor((K - 1) / O) + 1, with O the smallest integer s.t. (K - 1) / O < f."""
    num_iters = (num_images - 1) // num_frames + 1
    return (num_images - 1) // num_iters + 1


def resize_crop(images, height, width, mode='nearest'):
    """Scale (B, C, H, W) images to cover (height, width), then center-crop."""
    h, w = images.shape[-2:]
    ratio = max(height / h, width / w)
    size = (max(int(h * ratio), height), max(int(w * ratio), width))
    if mode == 'nearest':  # F.interpolate's default, as in the research code
        images = F.interpolate(images, size=size)
    else:
        images = F.interpolate(images, size=size, mode=mode, align_corners=False, antialias=True).clamp(0, 1)
    return CenterCrop((height, width))(images)


def find_mask(mask_dir, name):
    """Mask of an image: same file name in ``mask_dir``, or the same stem with another extension."""
    if mask_dir is None:
        return None
    candidates = [os.path.join(mask_dir, name)]
    stem = os.path.splitext(name)[0]
    candidates += [os.path.join(mask_dir, stem + ext) for ext in ('.png', '.PNG', '.jpg', '.JPG', '.jpeg')]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_view(image_path, mask_path, height, width, resize_mode='nearest'):
    """Load one view as an (H, W, 3) image in [0, 1] and an (H, W) inpainting mask in [0, 1].

    Mask convention: white (255) = transient pixels to inpaint, black (0) = keep. Lossless masks
    (e.g. PNG) may also store 0/1. Without a mask nothing is inpainted (all zeros).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Cannot read image {image_path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hw = image.shape[:2]
    image = torch.from_numpy(image / 255.).permute(2, 0, 1)
    image = resize_crop(image[None], height, width, resize_mode).squeeze(0).permute(1, 2, 0)
    if mask_path is None:
        return image, torch.zeros(height, width)

    mask = cv2.imread(mask_path)
    if mask is None:
        raise FileNotFoundError(f'Cannot read mask {mask_path}')
    if mask.shape[:2] != image_hw:
        mask_hw = mask.shape[:2]
        if abs(mask_hw[0] / mask_hw[1] - image_hw[0] / image_hw[1]) > 0.01:
            print(f'Warning: mask {mask_path} ({mask_hw[1]}x{mask_hw[0]}) has another aspect ratio than its '
                  f'image ({image_hw[1]}x{image_hw[0]}); it is stretched to the image size.')
        mask = cv2.resize(mask, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST)
    lossless = os.path.splitext(mask_path)[1].lower() in ('.png', '.bmp', '.tif', '.tiff')
    if lossless and mask.max() <= 1:  # 0/1 mask
        mask = mask.astype(np.float64)
    else:  # 0/255 mask
        if 1 < mask.max() < 128:
            print(f'Warning: mask {mask_path} has a maximum of {mask.max()}; masks are expected to be 0/255.')
        mask = mask / 255.
    mask = torch.from_numpy(mask)[:, :, 0]
    mask = resize_crop(mask[None, None].float(), height, width, resize_mode).squeeze()
    return image, mask


# ----------------------------------------------------------------------------------------
# Video diffusion model
# ----------------------------------------------------------------------------------------

def load_model_checkpoint(model, ckpt_path):
    """Load a pytorch-lightning (or deepspeed) checkpoint into the model."""
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'module' in state_dict:  # deepspeed
        state_dict = OrderedDict((k[16:], v) for k, v in state_dict['module'].items())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f'Warning: {len(missing)} missing and {len(unexpected)} unexpected keys in {ckpt_path}')
    print(f'>>> model checkpoint loaded from {ckpt_path}')
    return model


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    return rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)


def image_guided_synthesis(model, prompts, videos, noise_shape, mask, style_frame, n_samples=1, ddim_steps=50,
                           ddim_eta=1., unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False,
                           multiple_cond_cfg=False, timestep_spacing='uniform', guidance_rescale=0.0,
                           condition_index=None, verbose=False, **kwargs):
    """Sample a restored video.

    Args:
        videos: (1, 3, T, H, W) initial video in [-1, 1] (zero frames and masked pixels are -1).
        mask: (1, T, H, W) inpainting masks in [-1, 1] (+1 = inpaint).
        style_frame: frame index of the style image (style mask +1 there, -1 elsewhere).
        condition_index: frames whose CLIP embeddings condition the Multi-input Query Transformer.
    Returns:
        (1, n_samples, 3, T, H, W) decoded samples (nominally in [-1, 1], not clamped).
    """
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSamplerMultiCond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [''] * batch_size
    assert condition_index is not None, 'Error: condition index is None!'

    # Multi-input Query Transformer over the CLIP tokens of all conditioning views.
    img = videos[:, :, condition_index]
    img = rearrange(img, 'b c t h w -> (b t) c h w')
    img_emb = model.embedder(img)
    img_emb = rearrange(img_emb, '(b t) l c -> b (t l) c', b=1)
    img_emb = model.image_proj_model([img_emb.squeeze()])

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {'c_crossattn': [torch.cat([cond_emb, img_emb], dim=1)]}

    if model.model.conditioning_key == 'hybrid':
        # Concatenated condition: initial-video latents + inpainting mask + style mask.
        z = get_latent_z(model, videos)  # b c t h w
        mask = F.interpolate(mask, z.shape[-2:])
        mask_ref = torch.zeros_like(mask) - 1
        mask_ref[0, style_frame] = 1.
        img_cat_cond = torch.cat([z, mask.unsqueeze(1), mask_ref.unsqueeze(1)], dim=1)
        cond['c_concat'] = [img_cat_cond]

    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == 'empty_seq':
            uc_emb = model.get_learned_conditioning(batch_size * [''])
        elif model.uncond_type == 'zero_embed':
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img))
        uc_img_emb = rearrange(uc_img_emb, '(b t) l c -> b (t l) c', b=1)
        uc_img_emb = model.image_proj_model([uc_img_emb.squeeze()])
        uc = {'c_crossattn': [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc['c_concat'] = [img_cat_cond]
    else:
        uc = None

    # One more unconditional branch (image = yes, text = "") for multi-condition CFG.
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {'c_crossattn': [torch.cat([uc_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2['c_concat'] = [img_cat_cond]
        kwargs.update({'unconditional_conditioning_img_nonetext': uc_2})
    else:
        kwargs.update({'unconditional_conditioning_img_nonetext': None})

    batch_variants = []
    for _ in range(n_samples):
        samples, _ = ddim_sampler.sample(S=ddim_steps,
                                         conditioning=cond,
                                         batch_size=batch_size,
                                         shape=noise_shape[1:],
                                         verbose=verbose,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=uc,
                                         eta=ddim_eta,
                                         cfg_img=cfg_img,
                                         mask=None,
                                         x0=None,
                                         fs=fs,
                                         timestep_spacing=timestep_spacing,
                                         guidance_rescale=guidance_rescale,
                                         **kwargs)
        batch_variants.append(model.decode_first_stage(samples))
    # variants, batch, c, t, h, w -> batch, variants, c, t, h, w
    return torch.stack(batch_variants).permute(1, 0, 2, 3, 4, 5)


class UniVerse:
    """The mask- and style-conditioned video diffusion model that restores initial videos."""

    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device(opts.device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('UniVerse inference requires a CUDA GPU.')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        # The DDIM sampler registers its buffers on the *current* CUDA device.
        torch.cuda.set_device(self.device)
        if not os.path.exists(opts.ckpt):
            raise FileNotFoundError(f'Checkpoint not found: {opts.ckpt}')
        # Seed before building the model, as in the research code: the VAE posterior sampling
        # uses the CPU random generator, whose state depends on the model initialisation.
        seed_everything(opts.seed)

        config = OmegaConf.load(opts.config)
        model_config = config.pop('model', OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = opts.perframe_ae
        model = load_model_checkpoint(model, opts.ckpt)
        model.eval()
        self.model = model

        channels = model.model.diffusion_model.out_channels
        self.noise_shape = [1, channels, NUM_FRAMES, opts.height // 8, opts.width // 8]

    @torch.no_grad()
    def restore_video(self, video, mask, style_frame, condition_index):
        """Restore one initial video.

        Args:
            video: (T, H, W, 3) initial video in [0, 1].
            mask: (T, H, W) or (1, T, H, W) inpainting masks in [0, 1].
            style_frame: frame index of the style image.
            condition_index: frame indices of the input views (MiQT conditioning).
        Returns:
            (T, H, W, 3) restored video in [-1, 1] (float16 on the GPU, from autocast).
        """
        opts = self.opts
        videos = (video * 2. - 1.).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        if mask.dim() == 3:
            mask = mask[None]
        mask = mask.to(self.device) * 2. - 1.
        with torch.cuda.amp.autocast():
            samples = image_guided_synthesis(
                self.model, [opts.prompt], videos, self.noise_shape, mask, style_frame,
                n_samples=1, ddim_steps=opts.ddim_steps, ddim_eta=opts.ddim_eta,
                unconditional_guidance_scale=opts.cfg_scale, cfg_img=opts.cfg_img, fs=opts.frame_stride,
                text_input=True, multiple_cond_cfg=opts.multiple_cond_cfg,
                timestep_spacing=opts.timestep_spacing, guidance_rescale=opts.guidance_rescale,
                condition_index=condition_index, verbose=opts.verbose)
        return torch.clamp(samples[0][0].permute(1, 2, 3, 0), -1., 1.)


# ----------------------------------------------------------------------------------------
# Iterative restoration (Supp. Alg. 1)
# ----------------------------------------------------------------------------------------

def save_video(frames, path, fps=8):
    """Save a (T, H, W, 3) tensor in [0, 1] as an H.264 mp4."""
    frames = (frames.detach().cpu().float().clamp(0, 1) * 255).to(torch.uint8)
    torchvision.io.write_video(path, frames, fps=fps, video_codec='h264', options={'crf': '10'})


def image_save_path(path):
    """Output path of a restored image: the input name, plus '.png' if it has no image extension."""
    if os.path.splitext(path)[1].lower() not in Image.registered_extensions():
        path += '.png'
    return path


def save_image(image, path):
    """Save an (H, W, 3) tensor in [0, 1]; the format follows the file extension (JPEG at quality 95)."""
    array = (image.detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.splitext(path)[1].lower() in ('.jpg', '.jpeg'):
        Image.fromarray(array).save(path, quality=95)
    else:
        Image.fromarray(array).save(path)


def plan_iterations(num_views, split=None, style_index=0):
    """Check the restoration settings; returns (split, number of views in the first iteration)."""
    if num_views < 2:
        raise ValueError('UniVerse needs at least 2 input views.')
    if split is None:
        split = images_per_iteration(num_views) - 1
    if not 1 <= split <= NUM_FRAMES - 1:
        raise ValueError(f'split must be in [1, {NUM_FRAMES - 1}], got {split}')
    first_size = min(split + 1, num_views)
    if not 0 <= style_index < first_size:
        raise ValueError(f'The style image must be one of the first {first_size} views in trajectory order '
                         f'(the views of the first iteration), got index {style_index}.')
    return split, first_size


def restore_scene(universe, views, distances, mask_dir, out_dir, style_index=0, split=None,
                  notebook_compat=False, save_videos=True):
    """Iteratively restore all views (Supp. Alg. 1).

    The first iteration takes split + 1 views; every following iteration takes the last
    restored image (first frame and style reference) plus the next ``split`` views.

    Args:
        views: (image path, output name) pairs in trajectory order.
        distances: (K-1,) distances between consecutive views.
        style_index: which view of the first iteration is the style image.
        split: new views per iteration (N - 1); None = paper rule.
        notebook_compat: build the model inputs exactly like the research notebook. It differs
            from the paper in three places: (1) the first iteration conditions the Multi-input
            Query Transformer on a single view (the 7th) instead of all N, (2) the carried-over
            restored frame keeps an all-ones inpainting mask, (3) later iterations build masks on a
            576x1024 canvas.
    Returns:
        OrderedDict output name -> restored (H, W, 3) image in [0, 1], in trajectory order.
    """
    opts = universe.opts
    height, width = opts.height, opts.width
    num_views = len(views)
    split, first_size = plan_iterations(num_views, split, style_index)

    image_out_dir = os.path.join(out_dir, 'images')
    video_out_dir = os.path.join(out_dir, 'videos')
    os.makedirs(image_out_dir, exist_ok=True)
    if save_videos:
        os.makedirs(video_out_dir, exist_ok=True)

    restored, log = OrderedDict(), []
    start, last_restored, it = 0, None, 0
    while start < num_views:
        first = last_restored is None
        idx = list(range(0, first_size)) if first else list(range(start - 1, min(start + split, num_views)))
        positions = frame_positions(distances[idx[0]:idx[-1]])

        video = torch.zeros(NUM_FRAMES, height, width, 3)
        # Inpainting masks: zero frames are inpainted entirely (mask = 1).
        mask_hw = (576, 1024) if notebook_compat and not first else (height, width)
        masks = torch.ones(NUM_FRAMES, *mask_hw)
        for i, (view, pos) in enumerate(zip(idx, positions)):
            if not first and i == 0:
                video[pos] = last_restored
                if not notebook_compat:  # an already restored image: nothing to inpaint
                    masks[pos] = 0.
                continue
            image_path, name = views[view]
            image, mask = load_view(image_path, find_mask(mask_dir, name), height, width, opts.resize_mode)
            video[pos] = image * (1 - mask[..., None])  # masked pixels are set to zero
            masks[pos] = F.interpolate(mask.float()[None, None], size=mask_hw).squeeze()

        if first:
            style_frame = positions[style_index]
            # The research notebook conditioned its first iteration on one view only.
            cond_index = [positions[min(6, len(positions) - 1)]] if notebook_compat else positions
        else:
            style_frame, cond_index = 0, positions  # the last restored image is the style image

        names = [views[v][1] for v in idx]
        print(f'[iteration {it}] {len(idx)} views ({names[0]} ... {names[-1]}) at frames {positions}, '
              f'style frame {style_frame}')
        output = universe.restore_video(video, masks[None], style_frame, torch.tensor(cond_index))
        # (x + 1) / 2 is computed in fp16 on the GPU, exactly as in the research code; the cast
        # to float32 afterwards is lossless, so the frame carried to the next iteration is unchanged.
        output = ((output + 1.) / 2.).detach().cpu().float()

        if save_videos:
            prefix = os.path.join(video_out_dir, f'iter{it:02d}')
            save_video(video, f'{prefix}_input.mp4')
            save_video(masks[..., None].repeat(1, 1, 1, 3), f'{prefix}_inpaint_mask.mp4')
            style = torch.zeros(NUM_FRAMES, height, width, 3)
            style[style_frame] = 1.
            save_video(style, f'{prefix}_style_mask.mp4')
            save_video(output, f'{prefix}_restored.mp4')

        for i, (view, pos) in enumerate(zip(idx, positions)):
            if not first and i == 0:
                continue  # restored by the previous iteration
            name = views[view][1]
            restored[name] = output[pos].clone()
            path = image_save_path(os.path.join(image_out_dir, name))
            save_image(output[pos], path)
            log.append(f'{it}\t{pos}\t{os.path.relpath(path, image_out_dir)}')

        last_restored = output[positions[-1]].clone()
        start += first_size if first else split
        it += 1

    with open(os.path.join(out_dir, 'order.txt'), 'w') as f:
        f.write('# iteration\tframe\timage\n' + '\n'.join(log) + '\n')
    if save_videos:
        save_video(torch.stack(list(restored.values())), os.path.join(out_dir, 'restored.mp4'), fps=4)
    return restored


# ----------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------

def str2bool(x):
    if str(x).lower() in ('1', 'true', 'yes', 'on'):
        return True
    if str(x).lower() in ('0', 'false', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean, got {x}')


def get_parser():
    parser = argparse.ArgumentParser(description='UniVerse: restore inconsistent multi-view images.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    parser.add_argument('--image_dir', type=str, required=True, help='Directory of input images.')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help="Inpainting masks (white-on-black, 0/255) with the same file names as the images "
                             "or <stem>.png. Default: '<image_dir>/../masks' if it exists.")
    parser.add_argument('--colmap', type=str, default=None,
                        help="COLMAP sparse model dir or images.txt / images.bin. "
                             "Default: '<image_dir>/../sparse/0' if it exists.")
    parser.add_argument('--transforms', type=str, default=None,
                        help="NeRF-style transforms.json with camera poses. Default: '<image_dir>/../transforms.json' "
                             "if it exists and there is no COLMAP model.")
    parser.add_argument('--no_poses', action='store_true',
                        help='Ignore poses: keep the file-name order and space the views uniformly.')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory.')
    # restoration
    parser.add_argument('--style_index', type=int, default=0,
                        help='Style image: index in trajectory order; it must be a view of the first iteration '
                             '(any view when all views fit in one iteration, i.e. K <= 25).')
    parser.add_argument('--style_image', type=str, default=None,
                        help='Style image given by file name (overrides --style_index; same constraint).')
    parser.add_argument('--split', type=int, default=None,
                        help='New views per iteration (N - 1). Default: the paper rule N = floor((K-1)/O) + 1.')
    parser.add_argument('--notebook_compat', action='store_true',
                        help="Build the model inputs like the research notebook (see restore_scene). For the "
                             "notebook's fern run also pass --model 512 --split 14 --style_index 5.")
    parser.add_argument('--resize_mode', type=str, default='nearest', choices=['nearest', 'bilinear', 'bicubic'],
                        help='Interpolation used to resize the inputs to the model resolution.')
    parser.add_argument('--no_video', action='store_true',
                        help='Do not save the per-iteration videos and restored.mp4.')
    # model
    parser.add_argument('--model', type=str, default='1024', choices=list(MODEL_PRESETS),
                        help='576x1024 (stage 2, final) or 320x512 (stage 1, faster) model.')
    parser.add_argument('--config', type=str, default=None, help='Model config (default: from --model).')
    parser.add_argument('--ckpt', type=str, default=None, help='Model checkpoint (default: from --model).')
    parser.add_argument('--height', type=int, default=None, help='Default: from the config (multiple of 64).')
    parser.add_argument('--width', type=int, default=None, help='Default: from the config (multiple of 64).')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device.')
    # sampling
    parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM sampling steps.')
    parser.add_argument('--ddim_eta', type=float, default=1.0, help='DDIM eta (1 = stochastic sampling).')
    parser.add_argument('--cfg_scale', '--unconditional_guidance_scale', dest='cfg_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale.')
    parser.add_argument('--guidance_rescale', type=float, default=0.7, help='Rescaling of the guided prediction.')
    parser.add_argument('--timestep_spacing', type=str, default='uniform_trailing',
                        choices=['uniform', 'uniform_trailing', 'quad'], help='DDIM timestep spacing.')
    parser.add_argument('--frame_stride', type=int, default=10, help='fps condition of the video model.')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Text prompt of the video model.')
    parser.add_argument('--multiple_cond_cfg', action='store_true', help='Separate image / text guidance.')
    parser.add_argument('--cfg_img', type=float, default=None, help='Image guidance scale (--multiple_cond_cfg).')
    parser.add_argument('--perframe_ae', type=str2bool, default=True, help='Decode frame by frame to save memory.')
    parser.add_argument('--seed', type=int, default=3213, help='Random seed.')
    parser.add_argument('--verbose', action='store_true', help='Show the DDIM progress bar.')
    return parser


def resolve_model_args(opts):
    """Fill --config / --ckpt / --height / --width from the --model preset and the config."""
    preset = MODEL_PRESETS[opts.model]
    if opts.config is not None and opts.ckpt is None:
        print(f'Warning: --config given without --ckpt; using the {opts.model} preset checkpoint.')
    opts.config = opts.config or os.path.join(ROOT, preset['config'])
    opts.ckpt = opts.ckpt or os.path.join(ROOT, preset['ckpt'])
    if opts.height is None or opts.width is None:
        latent_h, latent_w = OmegaConf.load(opts.config).model.params.image_size
        opts.height = latent_h * 8 if opts.height is None else opts.height
        opts.width = latent_w * 8 if opts.width is None else opts.width
    if opts.height <= 0 or opts.width <= 0 or opts.height % 64 or opts.width % 64:
        raise ValueError(f'height and width must be positive multiples of 64, got {opts.height}x{opts.width}')
    return opts


def find_style_index(views, style_image):
    """Index (in trajectory order) of the view named ``style_image`` (full name or unique file name)."""
    names = [name for _, name in views]
    if style_image in names:
        return names.index(style_image)
    matches = [i for i, name in enumerate(names) if os.path.basename(name) == os.path.basename(style_image)]
    if len(matches) != 1:
        raise ValueError(f'--style_image {style_image} matches {len(matches)} of the views')
    return matches[0]


def main():
    opts = resolve_model_args(get_parser().parse_args())
    if not os.path.exists(opts.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {opts.ckpt} (see README, "Checkpoints")')
    views, distances, mask_dir = prepare_scene(opts.image_dir, opts.mask_dir, opts.colmap, opts.transforms,
                                               opts.no_poses)
    opts.mask_dir = mask_dir
    style_index = opts.style_index
    if opts.style_image is not None:
        style_index = find_style_index(views, opts.style_image)
    plan_iterations(len(views), opts.split, style_index)  # fail early, before loading the model

    image_out_dir = os.path.realpath(os.path.join(opts.out_dir, 'images'))
    if image_out_dir in (os.path.realpath(opts.image_dir), os.path.realpath(mask_dir or opts.image_dir)):
        raise ValueError(f'--out_dir {opts.out_dir} would overwrite the input images or masks')
    if not opts.no_video:
        try:
            import av  # noqa: F401  (torchvision's video writer)
        except ImportError:
            print('Warning: PyAV is not installed, videos are not saved (pip install av).')
            opts.no_video = True
    try:
        import xformers  # noqa: F401
    except ImportError:
        print('Warning: xformers is not installed; without memory-efficient attention UniVerse-1024 needs '
              'more than 48 GB of GPU memory (pip install xformers==0.0.16).')

    os.makedirs(opts.out_dir, exist_ok=True)
    with open(os.path.join(opts.out_dir, 'args.json'), 'w') as f:
        json.dump(vars(opts), f, indent=2)
    universe = UniVerse(opts)
    restore_scene(universe, views, distances, mask_dir, opts.out_dir, style_index=style_index,
                  split=opts.split, notebook_compat=opts.notebook_compat, save_videos=not opts.no_video)
    print(f'Done. Restored images are in {os.path.join(opts.out_dir, "images")}')


if __name__ == '__main__':
    main()

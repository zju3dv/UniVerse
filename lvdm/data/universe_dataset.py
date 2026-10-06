# Adapted from the WebVid dataset of DynamiCrafter (https://github.com/Doubiiu/DynamiCrafter) and
# ViewCrafter (https://github.com/Drexubery/ViewCrafter), Apache License 2.0.
# Modified for UniVerse: online synthesis of the inconsistent training pairs (get_deg_mask).
"""UniVerse training data: (inconsistent video, consistent video) pairs synthesised online.

Each sample starts from a clean 25-frame clip (``<data_dir>/clean_h/*.mp4``, written by
``scripts/prepare_dl3dv_clips.py``):

- N input frames are kept at random positions, always including the first and the last frame;
  the other frames are zero frames that the model has to generate (inpainting mask = 1);
- every input frame is degraded with ``lvdm.data.degradations.random_deg``; occluded pixels are
  added to the inpainting mask;
- one input frame is the style reference (``refer_num``): the target is the clean clip
  re-coloured with that frame's photometric jitter;
- with probability 0.15 the first 5 frames are kept clean (already consistent inputs).
"""
import os
import random

import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms

from lvdm.data.degradations import jitter, list_mask_files, random_deg

VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm')


def allocate_weights_to_integers(weights: torch.Tensor, total: int):
    """Split ``total`` into integers proportional to ``weights``; the remainder goes to the
    entries with the largest fractional parts."""
    normalized_weights = weights / weights.sum()
    scaled_values = normalized_weights * total
    integer_parts = torch.floor(scaled_values).to(torch.int32)
    remainder = total - integer_parts.sum()
    fractional_parts = scaled_values - integer_parts
    sorted_indices = torch.argsort(fractional_parts, descending=True)
    for i in range(remainder):
        integer_parts[sorted_indices[i]] += 1
    return integer_parts


class UniVerseDataset(Dataset):
    """Clean clips degraded online into UniVerse training pairs.

    Args:
        data_dir: directory that contains ``clip_subdir`` with the clean clips.
        mask_dirs: directories of occluder masks (> 0 = object, read by ``degradations.load_mask``), e.g.
            VOC2007 ``SegmentationClass``. A directory is chosen uniformly, then a file.
        resolution: (H, W) of the training frames, any size from 320x512 up (the blur kernel sizes
            scale with it); the paper trains at 320x512, then at 576x1024.
        video_length: frames per sample (the model is trained with 25).
        repeat: number of times the clip list is repeated in one epoch.
        clip_subdir: sub-directory of ``data_dir`` holding the clips.
        spatial_transform: 'resize_center_crop', 'center_crop', 'resize' or None.
        caption: text prompt of every sample.
        fps: frame-rate condition of every sample.
    """
    def __init__(self,
                 data_dir,
                 mask_dirs,
                 resolution=(576, 1024),
                 video_length=25,
                 repeat=2,
                 clip_subdir="clean_h",
                 spatial_transform="resize_center_crop",
                 caption="A consistent rotating view",
                 fps=10,
                 ):
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else [int(r) for r in resolution]
        if self.resolution[0] < 320 or self.resolution[1] < 512:
            raise ValueError(f"resolution (H, W) must be at least 320x512, got {self.resolution}")
        if video_length < 6:
            raise ValueError(f"video_length must be at least 6, got {video_length}")
        self.video_length = video_length
        self.caption = caption
        self.fps = fps

        if spatial_transform == "resize_center_crop":
            self.spatial_transform = transforms.Compose([
                transforms.Resize(min(self.resolution), antialias=True),
                transforms.CenterCrop(self.resolution),
            ])
        elif spatial_transform == "center_crop":
            self.spatial_transform = transforms.CenterCrop(self.resolution)
        elif spatial_transform == "resize":
            self.spatial_transform = transforms.Resize(self.resolution)
        elif spatial_transform is None:
            self.spatial_transform = None
        else:
            raise NotImplementedError(f"Unknown spatial_transform: {spatial_transform}")

        self.clip_dir = os.path.join(data_dir, clip_subdir)
        if not os.path.isdir(self.clip_dir):
            raise FileNotFoundError(f"Clip directory not found: {self.clip_dir}")
        clip_paths = sorted(os.path.join(self.clip_dir, name) for name in os.listdir(self.clip_dir)
                            if name.lower().endswith(VIDEO_EXTS))
        if not clip_paths:
            raise FileNotFoundError(f"No video clips found in {self.clip_dir}")
        self.clip_paths = clip_paths * repeat
        self.mask_pool = list_mask_files(mask_dirs)

    def get_deg_mask(self, frames):
        """Synthesise a training pair from clean uint8 frames (T, 3, H, W).

        Returns uint8 (T, 3, H, W) tensors: the target (re-styled to the style frame), the degraded
        input video (zero frames are black) and the inpainting mask (255 = generate / inpaint,
        0 = keep); plus the input frame indices (LongTensor, may hold one duplicate) and the style
        frame index.
        """
        num_frames = frames.shape[0]
        deg = torch.zeros_like(frames).float()
        mask = torch.ones_like(frames).float()
        clean_prefix = random.uniform(0, 1.) <= 0.15
        if not clean_prefix:
            # k + 1 inputs: frame 0 plus k frames spread with random gaps, the last one at T - 1
            d = torch.rand(random.choices(range(1, num_frames + 1), k=1)[0])
            gaps = allocate_weights_to_integers(d, num_frames - (d.shape[0] + 1)) + 1
            frameid = [0]
        else:
            # 5 clean inputs (frames 0-4) plus k degraded ones, the last one at T - 1
            d = torch.rand(random.choices(range(1, num_frames - 4), k=1)[0])
            gaps = allocate_weights_to_integers(d, num_frames - 5 - d.shape[0]) + 1
            frameid = [0, 1, 2, 3, 4]
        for gap in gaps:
            frameid.append((frameid[-1] + gap).item())
        refer_num = random.choice(frameid)

        frameid = torch.tensor(frameid)
        new_frames = frames.clone()
        for i in range(len(frameid)):
            if clean_prefix and i <= 4:
                deg_img = frames[frameid[i]] / 255.
                mask_now = torch.zeros_like(frames[frameid[i]])
            else:
                deg_img, mask_now, params = random_deg(frames[frameid[i]] / 255., self.mask_pool)
                if frameid[i] == refer_num:
                    new_frames = jitter.apply_deterministic(frames / 255., params).clip(0, 1.)
                    new_frames = (new_frames * 255.).clip(0, 255).to(torch.uint8)
            deg[frameid[i]] = deg_img
            mask[frameid[i]] = mask_now
        deg = (deg * 255.).clip(0, 255.).to(torch.uint8)
        mask = (mask * 255.).clip(0, 255.).to(torch.uint8)
        return new_frames, deg, mask, frameid, refer_num

    def __getitem__(self, index):
        ## get frames until success
        for _ in range(len(self.clip_paths)):
            video_path = self.clip_paths[index % len(self.clip_paths)]
            try:
                video_reader = VideoReader(video_path, ctx=cpu(0))
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length ({self.video_length}): {video_path}")
                    index += 1
                    continue
                frames = video_reader.get_batch(list(range(self.video_length)))
                break
            except Exception:
                print(f"Load video failed! path = {video_path}")
                index += 1
        else:
            raise RuntimeError(f"No readable clip with at least {self.video_length} frames in {self.clip_dir}")

        frames = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2)  # [t,c,h,w] uint8
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        frames, frames_cond, mask, frameid, refer_num = self.get_deg_mask(frames)
        frames = frames.permute(1, 0, 2, 3).float()  # [t,c,h,w] -> [c,t,h,w]
        frames_cond = frames_cond.permute(1, 0, 2, 3).float()
        mask = mask.permute(1, 0, 2, 3)[[0], :, :, :].float()
        assert (frames.shape[2], frames.shape[3]) == tuple(self.resolution), \
            f'frames={frames.shape}, self.resolution={self.resolution}'

        frames = (frames / 255. - 0.5) * 2
        frames_cond = (frames_cond / 255. - 0.5) * 2
        mask = (mask / 255. - 0.5) * 2  # +1 = generate / inpaint, -1 = keep
        data = {'video': frames, 'caption': self.caption, 'path': video_path, 'fps': self.fps, 'frame_stride': 1,
                'video_cond': frames_cond, 'frameid': frameid, 'mask': mask, 'refer_num': refer_num}  # c t h w
        return data

    def __len__(self):
        return len(self.clip_paths)

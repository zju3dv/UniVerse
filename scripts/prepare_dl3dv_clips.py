r"""Extract clean 25-frame training clips from DL3DV-10K videos.

For every scene video (``<dl3dv_root>/<subset>/<scene>/video.mp4`` or ``<dl3dv_root>/<scene>/video.mp4``)
this samples ``--clips_per_scene`` clips of 25 frames with a random frame stride (>= 2, to mimic
varying view densities), resizes them to cover 576x1024 (nearest neighbour), center-crops them and
writes them as H.264 mp4 files to ``<out_dir>/clean_h/<subset>_<scene>_<clip>_<copy>.mp4``. The
training dataset (``lvdm.data.universe_dataset.UniVerseDataset``) degrades these clips online.

File names, clip sampling (given ``--seed``) and the shard of a scene depend only on its path below
``--dl3dv_root``, so a re-run, e.g. after downloading more subsets, skips the finished scenes and
extracts only the new ones. Example with 4 parallel shards:

    for i in 0 1 2 3; do
        python scripts/prepare_dl3dv_clips.py --dl3dv_root /path/to/DL3DV-10K --out_dir data/train \
            --num_shards 4 --shard_id $i &
    done; wait
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import tempfile
import zlib
from collections import Counter

import torch
import torch.nn.functional as F
import torchvision
from decord import VideoReader, cpu
from torchvision.transforms import CenterCrop
from tqdm import tqdm

FPS = 6


def find_videos(root, video_name):
    """All scene videos under ``root`` (one or two directory levels deep), sorted."""
    paths = glob.glob(os.path.join(root, '*', '*', video_name)) + glob.glob(os.path.join(root, '*', video_name))
    return sorted(set(paths))


def scene_prefix(scene):
    """File-name prefix of a scene path relative to the DL3DV root, e.g. '1K/<hash>' -> '1K_<hash>'."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', scene)


def sample_clip_indices(num_video_frames, num_frames, rng, min_stride=2):
    """Frame indices of one clip: random stride in [min_stride, num_video_frames // num_frames - 2]
    and random start. Returns (indices, start, stride), or None if the video is too short."""
    max_stride = num_video_frames // num_frames - 2
    if max_stride < min_stride:
        return None
    stride = rng.randint(min_stride, max_stride)
    start = rng.randint(0, num_video_frames - stride * num_frames - 2)
    return [start + i * stride for i in range(num_frames)], start, stride


def resize_center_crop(frames, height, width):
    """Scale (T, C, H, W) uint8 frames to cover (height, width) with nearest interpolation, then center-crop."""
    h, w = frames.shape[-2:]
    ratio = max(height / h, width / w)
    size = (max(int(h * ratio), height), max(int(w * ratio), width))
    return CenterCrop((height, width))(F.interpolate(frames, size=size))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, got {value}')
    return number


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dl3dv_root', type=str, required=True, help='DL3DV-10K root with <subset>/<scene>/video.mp4')
    parser.add_argument('--out_dir', type=str, required=True, help='clips are written to <out_dir>/clean_h/')
    parser.add_argument('--clips_per_scene', type=positive_int, default=4, help='clips sampled from each scene video')
    parser.add_argument('--copies', type=positive_int, default=1,
                        help='identical files written per clip. The paper data used 5: with the dataset `repeat` of 2, '
                             'every clip was seen 10 times per epoch, each time with new online degradations. With '
                             '--copies 1, raise the dataset `repeat` instead (e.g. to 10) to get the same sampling '
                             'without the extra disk space.')
    parser.add_argument('--num_frames', type=positive_int, default=25, help='frames per clip')
    parser.add_argument('--height', type=int, default=576)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--video_name', type=str, default='video.mp4', help='file name of the scene videos')
    parser.add_argument('--num_shards', type=positive_int, default=1,
                        help='split the scenes into this many shards (by a hash of the scene path)')
    parser.add_argument('--shard_id', type=int, default=0, help='shard processed by this run (0-based)')
    parser.add_argument('--seed', type=int, default=0, help='clip sampling seed (per scene, independent of sharding)')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error('--shard_id must be in [0, num_shards)')
    videos = find_videos(args.dl3dv_root, args.video_name)
    if not videos:
        raise FileNotFoundError(f'No {args.video_name} found under {args.dl3dv_root}')
    scenes = [os.path.relpath(os.path.dirname(path), args.dl3dv_root) for path in videos]
    prefixes = [scene_prefix(scene) for scene in scenes]
    clashes = sorted(prefix for prefix, count in Counter(prefixes).items() if count > 1)
    if clashes:
        raise ValueError(f'Different scene paths map to the same clip names: {clashes[:5]}')
    clip_dir = os.path.join(args.out_dir, 'clean_h')
    os.makedirs(clip_dir, exist_ok=True)
    todo = [i for i, scene in enumerate(scenes) if zlib.crc32(scene.encode()) % args.num_shards == args.shard_id]
    print(f'{len(videos)} scene videos, {len(todo)} in shard {args.shard_id}/{args.num_shards}')

    manifest_path = os.path.join(args.out_dir, f'manifest_{args.shard_id:03d}.jsonl')
    num_written = 0
    # clips are written here and then moved into clip_dir, so an interrupted run leaves no partial clip
    tmp_dir = tempfile.mkdtemp(prefix='.tmp_', dir=clip_dir)
    try:
        for i in tqdm(todo):
            path, scene = videos[i], scenes[i]
            names = [[f'{prefixes[i]}_{clip:02d}_{copy:02d}.mp4' for copy in range(args.copies)]
                     for clip in range(args.clips_per_scene)]
            if all(os.path.exists(os.path.join(clip_dir, name)) for clip_names in names for name in clip_names):
                continue
            try:
                vr = VideoReader(path, ctx=cpu(0))
            except Exception as e:
                print(f'Skipping {path}: cannot read video ({e})')
                continue
            rng = random.Random(f'{args.seed}/{scene}')
            entries = []
            for clip in range(args.clips_per_scene):
                sample = sample_clip_indices(len(vr), args.num_frames, rng)
                if sample is None:
                    print(f'Skipping {path}: {len(vr)} frames, at least {(2 + 2) * args.num_frames} are needed')
                    break
                indices, start, stride = sample
                frames = torch.from_numpy(vr.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)  # T C H W, RGB uint8
                frames = resize_center_crop(frames, args.height, args.width).permute(0, 2, 3, 1).contiguous()
                for copy, name in enumerate(names[clip]):
                    tmp_path = os.path.join(tmp_dir, name)
                    if copy == 0:
                        torchvision.io.write_video(tmp_path, frames, fps=FPS, video_codec='h264', options={'crf': '10'})
                    else:
                        shutil.copyfile(os.path.join(clip_dir, names[clip][0]), tmp_path)
                    os.replace(tmp_path, os.path.join(clip_dir, name))
                    entries.append(json.dumps({'clip': name, 'scene': scene, 'start': start, 'stride': stride}) + '\n')
            if entries:  # listed once the scene is complete, so a resumed scene is not listed twice
                with open(manifest_path, 'a') as manifest:
                    manifest.writelines(entries)
                num_written += len(entries)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f'Wrote {num_written} clips to {clip_dir}')


if __name__ == '__main__':
    main()

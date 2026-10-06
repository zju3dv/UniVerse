# Adapted from DynamiCrafter (https://github.com/Doubiiu/DynamiCrafter, utils/save_video.py), Apache License 2.0.
# Modified for UniVerse: only the helpers used by callbacks.py.
import os
import numpy as np
from PIL import Image

import torch
import torchvision


def log_local(batch_logs, save_dir, filename, save_fps=10, rescale=True):
    """ save images and videos from images dict """
    if batch_logs is None:
        return None

    def save_img_grid(grid, path, rescale):
        if rescale:
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid * 255).astype(np.uint8)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid).save(path)

    for key in batch_logs:
        value = batch_logs[key]
        if isinstance(value, list) and isinstance(value[0], str):
            ## a batch of captions
            path = os.path.join(save_dir, "%s-%s.txt" % (key, filename))
            with open(path, 'w') as f:
                for i, txt in enumerate(value):
                    f.write(f'idx={i}, txt={txt}\n')
        elif isinstance(value, torch.Tensor) and value.dim() == 5:
            ## save video grids
            video = value  # b,c,t,h,w
            ## only save grayscale or rgb mode
            if video.shape[1] != 1 and video.shape[1] != 3:
                continue
            video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
            frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(1), padding=0) for framesheet in video]  # [3, n*h, 1*w]
            grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
            if rescale:
                grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
            path = os.path.join(save_dir, "%s-%s.mp4" % (key, filename))
            torchvision.io.write_video(path, grid, fps=save_fps, video_codec='h264', options={'crf': '10'})
        elif isinstance(value, torch.Tensor) and value.dim() == 4:
            ## save image grids
            img = value
            ## only save grayscale or rgb mode
            if img.shape[1] != 1 and img.shape[1] != 3:
                continue
            grid = torchvision.utils.make_grid(img, nrow=1, padding=0)
            path = os.path.join(save_dir, "%s-%s.jpg" % (key, filename))
            save_img_grid(grid, path, rescale)
        else:
            pass


def prepare_to_log(batch_logs, max_images=100000, clamp=True):
    """ keep at most max_images samples per key, move tensors to CPU and clamp them to [-1, 1] """
    if batch_logs is None:
        return None
    for key in batch_logs:
        N = batch_logs[key].shape[0] if hasattr(batch_logs[key], 'shape') else len(batch_logs[key])
        N = min(N, max_images)
        batch_logs[key] = batch_logs[key][:N]
        ## in batch_logs: images <batched tensor> & caption <text list>
        if isinstance(batch_logs[key], torch.Tensor):
            batch_logs[key] = batch_logs[key].detach().cpu()
            if clamp:
                try:
                    batch_logs[key] = torch.clamp(batch_logs[key].float(), -1., 1.)
                except RuntimeError:
                    print("clamp_scalar_cpu not implemented for Half")
    return batch_logs

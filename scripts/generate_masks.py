"""Generate the inpainting masks of transient objects (people, cars, ...) for UniVerse.

inference.py inpaints the white pixels (255) of ``<scene>/masks/<image stem>.png`` and keeps the black ones (0);
views without a mask are not inpainted. This script writes such a mask for every image of a folder:

  gsam         (default) Grounded-SAM: GroundingDINO (Swin-T OGC) detects boxes for each text prompt and SAM
               (ViT-H) segments every box; masks covering at most --min_area of the image are dropped.
               About 8 GB of GPU memory and under a second per 1024x576 image (0.5-2 minutes with --device cpu).
  segnext_sam  The recipe of the paper: SegNeXt-L (ADE20K) segments the prompted classes and --num_points random
               pixels of them prompt SAM as positive points. Needs MMSegmentation 1.x.
  empty        All-zero masks: nothing is inpainted.

The masks of all prompts are merged, dilated with a --dilate x --dilate square kernel and saved as single-channel
0/255 PNGs. --dilate is in pixels of the input images: the default 5 matches the authors' masks of 1024x576
images, so scale it for larger images. --prompts are free text for gsam and ADE20K class names for segnext_sam;
the paper's example is ``person car bicycle`` (its "bike" is ``bicycle`` or ``minibike`` in ADE20K).

Setup (put the weights in the checkpoints/ folder of the repository, or pass their paths):
  gsam: GroundingDINO and segment_anything as vendored in Grounded-Segment-Anything, tested at this commit:
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
    cd Grounded-Segment-Anything && git checkout 126abe633ffe333e16e4a0a4e946bc1003caf757
    pip install -e segment_anything && pip install --no-build-isolation -e GroundingDINO  # with CUDA_HOME set
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    GroundingDINO also fetches bert-base-uncased from the Hugging Face hub on first use.
  segnext_sam (plus segment_anything and the SAM weights above; from the repository root):
    pip install -U openmim && mim install mmengine "mmcv>=2.0.0rc4,<2.2.0"
    pip install mmsegmentation==1.2.2 ftfy regex
    mim download mmsegmentation --config segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512 --dest checkpoints

Example:
    python scripts/generate_masks.py --images data/my_scene/images --prompts person car bicycle
    python inference.py --image_dir data/my_scene/images  # reads data/my_scene/masks
"""

import argparse
import os
import random
import sys
import time
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')  # as in inference.py
GDINO_URL = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'
SAM_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
SEGNEXT_URL = ('https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-l_1x16_512x512_adamw_160k_'
               'ade20k/segnext_mscan-l_1x16_512x512_adamw_160k_ade20k_20230209_172055-19b14b63.pth')
# Tested: GroundingDINO and segment_anything as vendored in Grounded-Segment-Anything at this commit.
GSA_SETUP = ('  git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git\n'
             '  cd Grounded-Segment-Anything && git checkout 126abe633ffe333e16e4a0a4e946bc1003caf757\n')
GDINO_INSTALL = GSA_SETUP + '  pip install --no-build-isolation -e GroundingDINO  # with CUDA_HOME set'
SAM_INSTALL = GSA_SETUP + '  pip install -e segment_anything'
MMSEG_INSTALL = ('pip install -U openmim && mim install mmengine "mmcv>=2.0.0rc4,<2.2.0" && '
                 'pip install mmsegmentation==1.2.2 ftfy regex')


def list_images(image_dir):
    return sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS))


def read_rgb(path):
    """Read an image as RGB uint8 exactly like inference.py (cv2, EXIF orientation applied)."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'Cannot read image {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def check_file(path, what, hint):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{what} not found: {path}. {hint}')


def check_sam(ckpt):
    """Check that segment_anything and the SAM weights are available, before any model is loaded."""
    try:
        from segment_anything import SamPredictor, sam_model_registry  # noqa: F401
    except ImportError as e:
        raise ImportError(f'segment_anything could not be imported ({e}). Install it with:\n{SAM_INSTALL}') from e
    check_file(ckpt, 'SAM checkpoint', f'Download it from {SAM_URL} (ViT-H) or pass --sam_ckpt.')


def build_sam(sam_type, ckpt, device):
    from segment_anything import SamPredictor, sam_model_registry
    print(f'Loading SAM ({sam_type}) from {ckpt}')
    return SamPredictor(sam_model_registry[sam_type](checkpoint=ckpt).to(device))


# Adapted from Grounded-Segment-Anything (https://github.com/IDEA-Research/Grounded-Segment-Anything,
# grounded_sam_demo.py), Apache License 2.0.
# Modified for UniVerse.
class GroundedSAM:
    """GroundingDINO boxes for each text prompt, segmented by SAM box prompts."""

    def __init__(self, opts):
        try:
            import groundingdino
            import groundingdino.datasets.transforms as T
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        except ImportError as e:
            raise ImportError(f'GroundingDINO could not be imported ({e}). Install it with:\n{GDINO_INSTALL}') from e
        if torch.device(opts.device).type == 'cuda':
            try:
                from groundingdino import _C  # noqa: F401
            except ImportError as e:
                raise ImportError('GroundingDINO was built without its CUDA op: reinstall it with CUDA_HOME set, '
                                  'or use --device cpu.') from e
        config = opts.gdino_config or os.path.join(os.path.dirname(groundingdino.__file__), 'config',
                                                     'GroundingDINO_SwinT_OGC.py')
        check_file(config, 'GroundingDINO config', 'Pass --gdino_config <Grounded-Segment-Anything>/GroundingDINO/'
                   'groundingdino/config/GroundingDINO_SwinT_OGC.py.')
        check_file(opts.gdino_ckpt, 'GroundingDINO checkpoint', f'Download it from {GDINO_URL} or pass --gdino_ckpt.')
        check_sam(opts.sam_ckpt)
        print(f'Loading GroundingDINO from {opts.gdino_ckpt}')
        args = SLConfig.fromfile(config)
        args.device = opts.device
        model = build_model(args)
        checkpoint = torch.load(opts.gdino_ckpt, map_location='cpu')
        missing, _ = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        if missing:
            print(f'Warning: {len(missing)} missing keys in {opts.gdino_ckpt}')
        self.gdino = model.eval().to(opts.device)
        self.get_phrases = get_phrases_from_posmap
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),  # one size: deterministic
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.sam = build_sam(opts.sam_type, opts.sam_ckpt, opts.device)
        self.device, self.prompts, self.min_area = opts.device, opts.prompts, opts.min_area
        self.box_threshold, self.text_threshold = opts.box_threshold, opts.text_threshold

    @torch.no_grad()
    def detect(self, image, prompt):
        """GroundingDINO boxes (absolute xyxy), scores and phrases of one text prompt."""
        caption = prompt.lower().strip()
        if not caption.endswith('.'):
            caption += '.'
        inputs = self.transform(Image.fromarray(image), None)[0].to(self.device)
        outputs = self.gdino(inputs[None], captions=[caption])
        logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (num_queries, 256)
        boxes = outputs['pred_boxes'].cpu()[0]  # (num_queries, 4), normalized cxcywh
        keep = logits.max(dim=1)[0] > self.box_threshold
        logits, boxes = logits[keep], boxes[keep]
        tokenized = self.gdino.tokenizer(caption)
        phrases = [self.get_phrases(logit > self.text_threshold, tokenized, self.gdino.tokenizer) for logit in logits]
        h, w = image.shape[:2]
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes, logits.max(dim=1)[0], phrases

    @torch.no_grad()
    def __call__(self, image, name):
        """(H, W) bool mask of all prompted objects and the SAM prompts used, for an RGB uint8 image."""
        h, w = image.shape[:2]
        mask, detections, encoded = np.zeros((h, w), bool), [], False
        for prompt in self.prompts:
            boxes, scores, phrases = self.detect(image, prompt)
            if len(boxes) == 0:
                continue
            if not encoded:  # the SAM image encoder runs once per image, and only if something was detected
                self.sam.set_image(image)
                encoded = True
            sam_boxes = self.sam.transform.apply_boxes_torch(boxes, (h, w)).to(self.device)
            masks, _, _ = self.sam.predict_torch(point_coords=None, point_labels=None, boxes=sam_boxes,
                                                 multimask_output=False)
            for m, box, score, phrase in zip(masks[:, 0].cpu().numpy(), boxes.tolist(), scores.tolist(), phrases):
                kept = m.mean() > self.min_area  # drop tiny masks before merging
                if kept:
                    mask |= m
                detections.append(dict(box=box, label=f'{phrase} {score:.2f}', kept=kept))
        return mask, detections


class SegNeXtSAM:
    """The paper's recipe: SegNeXt-L (ADE20K) segments the prompted classes, random pixels of them prompt SAM."""

    def __init__(self, opts):
        try:
            from mmseg.apis import inference_model, init_model
        except ImportError as e:
            raise ImportError(f'The segnext_sam backend needs MMSegmentation 1.x ({e}). Install it with:\n'
                              f'  {MMSEG_INSTALL}') from e
        check_file(opts.segnext_config, 'SegNeXt config', 'Get it with: mim download mmsegmentation --config '
                   f'segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512 --dest {CKPT_DIR}')
        check_file(opts.segnext_ckpt, 'SegNeXt checkpoint', f'Download it from {SEGNEXT_URL}')
        check_sam(opts.sam_ckpt)
        print(f'Loading SegNeXt from {opts.segnext_ckpt}')
        self.segnext = init_model(opts.segnext_config, opts.segnext_ckpt, device=opts.device)
        self.inference_model = inference_model
        classes = [c.strip() for c in self.segnext.dataset_meta['classes']]
        prompts = [p.strip().lower() for p in opts.prompts]
        unknown = [p for p in prompts if p not in classes]
        if unknown:
            raise ValueError(f'{unknown} are not classes of the segmentation model. Choose from: {", ".join(classes)}')
        self.class_ids = [classes.index(p) for p in prompts]
        self.sam = build_sam(opts.sam_type, opts.sam_ckpt, opts.device)
        self.num_points, self.seed = opts.num_points, opts.seed

    @torch.no_grad()
    def __call__(self, image, name):
        """(H, W) bool mask of the prompted classes and the SAM points used, for an RGB uint8 image."""
        torch.manual_seed(self.seed)  # the Hamburger decode head draws random bases at every forward
        result = self.inference_model(self.segnext, np.ascontiguousarray(image[..., ::-1]))  # BGR input
        semantic = np.isin(result.pred_sem_seg.data[0].cpu().numpy(), self.class_ids)
        ys, xs = np.nonzero(semantic)
        if len(xs) < self.num_points:  # too few pixels to sample from: keep the semantic mask
            return semantic, []
        idx = random.Random(f'{self.seed}/{name}').sample(range(len(xs)), self.num_points)
        points = np.stack([xs[idx], ys[idx]], axis=1)
        self.sam.set_image(image)
        masks, _, _ = self.sam.predict(point_coords=points, point_labels=np.ones(len(points), dtype=int),
                                       multimask_output=False)
        return masks[0], [dict(point=p, label='', kept=True) for p in points.tolist()]


def write_mask(path, mask):
    """Write a 0/255 uint8 mask; the format follows the extension (JPEG only with --same_name)."""
    params = [cv2.IMWRITE_JPEG_QUALITY, 100] if path.lower().endswith(('.jpg', '.jpeg')) else []
    if not cv2.imwrite(path, mask, params):
        raise IOError(f'Cannot write {path}')


def save_overlay(path, image, mask, detections):
    """Save the image with the mask in red and the SAM prompts (green: kept, gray: dropped by --min_area)."""
    vis = image.copy()
    region = mask > 0
    vis[region] = (vis[region] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    for det in detections:
        color = (0, 255, 0) if det['kept'] else (160, 160, 160)
        if 'box' in det:
            x0, y0, x1, y1 = (int(round(v)) for v in det['box'])
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv2.putText(vis, det['label'], (x0 + 2, max(y0 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
        else:
            cv2.circle(vis, tuple(int(v) for v in det['point']), 5, color, -1)
    cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def get_parser():
    parser = argparse.ArgumentParser(description='Generate inpainting masks of transient objects for UniVerse.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images', type=str, required=True, help='Directory of input images.')
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory. Default: '<images>/../masks'.")
    parser.add_argument('--backend', type=str, default='gsam', choices=['gsam', 'segnext_sam', 'empty'],
                        help='Grounded-SAM, the paper recipe SegNeXt + SAM, or all-zero masks.')
    parser.add_argument('--prompts', type=str, nargs='+', default=['person'],
                        help='Transient objects: text prompts (gsam) or ADE20K class names (segnext_sam). '
                             'The paper uses: person car bicycle.')
    parser.add_argument('--dilate', type=int, default=5,
                        help='Size of the square dilation kernel in pixels of the input images (0 = none; odd sizes '
                             'keep it centered).')
    parser.add_argument('--same_name', action='store_true',
                        help='Name each mask exactly like its image (e.g. IMG_1.JPG, lossy) instead of <stem>.png.')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate masks that already exist.')
    parser.add_argument('--save_vis', action='store_true', help="Save overlays to '<out_dir>/vis'.")
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device of the segmentation models.')
    # gsam
    parser.add_argument('--box_threshold', type=float, default=0.3, help='GroundingDINO box score threshold.')
    parser.add_argument('--text_threshold', type=float, default=0.25,
                        help='GroundingDINO token threshold of the detected phrases (labels only).')
    parser.add_argument('--min_area', type=float, default=0.003,
                        help='Drop object masks covering at most this fraction of the image (gsam), in [0, 1).')
    parser.add_argument('--gdino_config', type=str, default=None,
                        help='GroundingDINO config. Default: GroundingDINO_SwinT_OGC.py of the installed package.')
    parser.add_argument('--gdino_ckpt', type=str, default=os.path.join(CKPT_DIR, 'groundingdino_swint_ogc.pth'),
                        help='GroundingDINO Swin-T OGC weights.')
    # SAM
    parser.add_argument('--sam_ckpt', type=str, default=os.path.join(CKPT_DIR, 'sam_vit_h_4b8939.pth'),
                        help='SAM weights.')
    parser.add_argument('--sam_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM backbone of --sam_ckpt.')
    # segnext_sam
    parser.add_argument('--segnext_config', type=str,
                        default=os.path.join(CKPT_DIR, 'segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512.py'),
                        help='SegNeXt-L ADE20K config (MMSegmentation 1.x).')
    parser.add_argument('--segnext_ckpt', type=str, default=os.path.join(CKPT_DIR, os.path.basename(SEGNEXT_URL)),
                        help='SegNeXt-L ADE20K weights.')
    parser.add_argument('--num_points', type=int, default=4, help='Random positive SAM points per image (segnext_sam).')
    parser.add_argument('--seed', type=int, default=0, help='Seed of the point sampling (segnext_sam).')
    return parser


def main():
    opts = get_parser().parse_args()
    if opts.dilate < 0 or opts.num_points < 1 or not 0 <= opts.min_area < 1:
        sys.exit('Error: --dilate must be >= 0, --num_points >= 1 and --min_area in [0, 1)')
    if opts.dilate > 0 and opts.dilate % 2 == 0:
        print(f'Warning: --dilate {opts.dilate} is even, so the dilation is off-center by half a pixel; '
              'odd sizes are centered')
    image_dir = os.path.normpath(opts.images)
    if not os.path.isdir(image_dir):
        sys.exit(f'Error: {image_dir} is not a directory')
    names = list_images(image_dir)
    if not names:
        sys.exit(f'Error: no images in {image_dir}')
    out_dir = opts.out_dir or os.path.join(os.path.dirname(image_dir), 'masks')
    if os.path.isdir(out_dir) and os.path.samefile(out_dir, image_dir):
        sys.exit('Error: --out_dir must differ from --images')
    out_names = [name if opts.same_name else os.path.splitext(name)[0] + '.png' for name in names]
    clashes = [name for name, count in Counter(out_names).items() if count > 1]
    if clashes:
        sys.exit(f'Error: several images would share the masks {clashes[:3]}; use --same_name')
    todo = [(name, out_name) for name, out_name in zip(names, out_names)
            if opts.overwrite or not os.path.exists(os.path.join(out_dir, out_name))]
    print(f'{len(names)} images in {image_dir}; writing {opts.backend} masks to {out_dir}')

    segmenter = None
    if todo and opts.backend != 'empty':
        device = torch.device(opts.device)
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)  # GroundingDINO's CUDA op runs on the current device
        try:
            segmenter = (GroundedSAM if opts.backend == 'gsam' else SegNeXtSAM)(opts)
        except (ImportError, FileNotFoundError, ValueError) as e:
            sys.exit(f'Error: {e}')
    os.makedirs(out_dir, exist_ok=True)
    if opts.save_vis:
        os.makedirs(os.path.join(out_dir, 'vis'), exist_ok=True)

    start = time.time()
    for i, (name, out_name) in enumerate(todo):
        t0 = time.time()
        image = read_rgb(os.path.join(image_dir, name))
        if segmenter is None:
            mask, detections = np.zeros(image.shape[:2], bool), []
        else:
            mask, detections = segmenter(image, name)
        mask = mask.astype(np.uint8) * 255
        if opts.dilate > 0:
            mask = cv2.dilate(mask, np.ones((opts.dilate, opts.dilate), np.uint8))
        write_mask(os.path.join(out_dir, out_name), mask)
        if opts.save_vis:
            save_overlay(os.path.join(out_dir, 'vis', os.path.splitext(name)[0] + '.jpg'), image, mask, detections)
        info = f', {sum(d["kept"] for d in detections)}/{len(detections)} SAM prompts kept' if detections else ''
        print(f'[{i + 1}/{len(todo)}] {name}: {(mask > 0).mean():.1%} masked{info} ({time.time() - t0:.2f}s)')

    print(f'Done: {len(todo)} masks written to {out_dir} in {time.time() - start:.1f}s')
    if len(todo) < len(names):
        print(f'{len(names) - len(todo)} existing masks were kept (use --overwrite to regenerate them)')
    shadowed = [name for name, out_name in zip(names, out_names)
                if out_name != name and os.path.exists(os.path.join(out_dir, name))]
    if shadowed:
        print(f'Warning: {out_dir} also contains masks named like the images (e.g. {shadowed[:3]}); inference.py '
              'reads those before <stem>.png. Remove them, or regenerate them with --same_name --overwrite.')


if __name__ == '__main__':
    main()

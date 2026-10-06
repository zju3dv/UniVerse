# Demo scene

Five of the 41 views of the *room* scene from the LLFF forward-facing data released with
[NeRF](https://github.com/bmild/nerf) (Mildenhall et al.), resized and center-cropped to 1024x576 and
synthetically degraded as in the paper's LLFF benchmark:

| Image | Degradation |
|---|---|
| `DJI_20200226_143850_006.JPG` | none (clean view) |
| `DJI_20200226_143902_603.JPG` | pasted PASCAL VOC 2007 object (parrot) + colour cast |
| `DJI_20200226_143913_463.JPG` | pasted PASCAL VOC 2007 object (train) + colour cast |
| `DJI_20200226_143933_787.JPG` | strong over-exposure / colour cast |
| `DJI_20200226_143946_704.JPG` | warm colour cast |

`masks/` marks the pasted objects (white = inpaint), and `sparse/0` is the scene's COLMAP model reduced
to these five views (2D points removed).

These images are third-party data used for demonstration only and are not covered by the repository
license. The pasted objects come from [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/),
whose images are subject to the terms of their original source (Flickr).

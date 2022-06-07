import torch
import logging
from pytorch_lightning.utilities import rank_zero_only

def to_coords(x: torch.Tensor, t: torch.Tensor):
    """
    Transforms the coordinates to a tensor X of shape [time, space, 2].
    Args:
        x: spatial coordinates
        t: temporal coordinates
    Returns:
        torch.Tensor: X[..., 0] is the space coordinate (in 2D)
                      X[..., 1] is the time coordinate (in 2D)
    """
    x_, t_ = torch.meshgrid(x, t)
    x_, t_ = x_.T, t_.T
    return torch.stack((x_, t_), -1)

def make_coord(shape, ranges=None, flatten=True):
    """ 
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
def get_logger(name=__name__):
    """
    Initializes multi-GPU-friendly python command line logger.
    https://github.com/ashleve/lightning-hydra-template/blob/8b62eef9d0d9c863e88c0992595688d6289d954f/src/utils/utils.py#L12
    """

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (C, L)
    """
    coord = make_coord(img.shape[-1:])
    rgb = img.view(img.shape[0], -1).permute(1, 0)
    return coord, rgb
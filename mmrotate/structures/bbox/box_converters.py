# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes, register_box_converter
from torch import Tensor

from .quadri_boxes import QuadriBoxes
from .rotated_boxes import RotatedBoxes


@register_box_converter(HorizontalBoxes, RotatedBoxes)
def hbox2rbox(boxes: Tensor) -> Tensor:
    """Convert horizontal boxes to rotated boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    wh = boxes[..., 2:] - boxes[..., :2]
    ctrs = (boxes[..., 2:] + boxes[..., :2]) / 2
    theta = boxes.new_zeros((*boxes.shape[:-1], 1))
    return torch.cat([ctrs, wh, theta], dim=-1)


@register_box_converter(HorizontalBoxes, QuadriBoxes)
def hbox2qbox(boxes: Tensor) -> Tensor:
    """Convert horizontal boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    x1, y1, x2, y2 = torch.split(boxes, 1, dim=-1)
    return torch.cat([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1)


@register_box_converter(RotatedBoxes, HorizontalBoxes)
def rbox2hbox(boxes: Tensor) -> Tensor:
    """Convert rotated boxes to horizontal boxes.

    Args:
        boxes (Tensor): Rotated box tensor with shape of (..., 5).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    ctrs, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * cos_value) + torch.abs(h / 2 * sin_value)
    y_bias = torch.abs(w / 2 * sin_value) + torch.abs(h / 2 * cos_value)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([ctrs - bias, ctrs + bias], dim=-1)


@register_box_converter(RotatedBoxes, QuadriBoxes)
def rbox2qbox(boxes: Tensor) -> Tensor:
    """Convert rotated boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): Rotated box tensor with shape of (..., 5).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    ctr, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
    vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return torch.cat([pt1, pt2, pt3, pt4], dim=-1)


@register_box_converter(QuadriBoxes, HorizontalBoxes)
def qbox2hbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to horizontal boxes.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    boxes = boxes.view(*boxes.shape[:-1], 4, 2)
    x1y1, _ = boxes.min(dim=-2)
    x2y2, _ = boxes.max(dim=-2)
    return torch.cat([x1y1, x2y2], dim=-1)


@register_box_converter(QuadriBoxes, RotatedBoxes)
def qbox2rbox(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to rotated boxes.

    Labeling convention: pts[0] and pts[1] are the two corners of the
    "top" edge (e.g. bottle cap end); pts[2] and pts[3] are the two
    corners of the opposite "bottom" edge (e.g. bottle base end).
    The resulting angle (in radians, range (-pi, pi]) points from the
    center toward the cap end, preserving full 360-degree orientation.

    Args:
        boxes (Tensor): Quadrilateral box tensor with shape of (..., 8).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    original_shape = boxes.shape[:-1]
    points = boxes.cpu().numpy().reshape(-1, 4, 2)
    rboxes = []
    for pts in points:
        cx, cy = pts.mean(axis=0)

        cap_mid = (pts[0] + pts[1]) / 2   # midpoint of cap (top) edge
        base_mid = (pts[2] + pts[3]) / 2  # midpoint of base (bottom) edge

        # Full 360-degree angle pointing from center toward cap end
        angle = np.arctan2(cap_mid[1] - cy, cap_mid[0] - cx)

        # w = length along the bottle axis (cap to base)
        w = float(np.linalg.norm(cap_mid - base_mid))
        # h = width across the bottle (length of the cap/base edge)
        h = float((np.linalg.norm(pts[1] - pts[0]) +
                   np.linalg.norm(pts[3] - pts[2])) / 2)

        rboxes.append([cx, cy, w, h, angle])
    rboxes = boxes.new_tensor(rboxes)
    return rboxes.view(*original_shape, 5)

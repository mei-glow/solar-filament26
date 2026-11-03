from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .refiner_model import cldice_loss, dice_loss


def build_semantic_model(architecture: str, encoder: str, encoder_weights: str | None = "imagenet"):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise RuntimeError("Install segmentation-models-pytorch from requirements.txt") from error
    kwargs = dict(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=3)
    if architecture == "unet":
        return smp.Unet(**kwargs)
    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(**kwargs)
    if architecture == "iterative_unetplusplus":
        base = smp.UnetPlusPlus(**kwargs)
        expert = smp.Unet(
            encoder_name="resnet18", encoder_weights=None, in_channels=4, classes=3
        )
        return IterativeResidualSemanticModel(base, expert)
    raise ValueError(f"Unknown architecture: {architecture}")


class IterativeResidualSemanticModel(nn.Module):
    """A full model followed by a light residual expert conditioned on its mask."""

    def __init__(self, base: nn.Module, expert: nn.Module):
        super().__init__()
        self.base = base
        self.expert = expert

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        initial = self.base(image)
        residual_input = torch.cat([image, initial[:, 0:1].sigmoid()], dim=1)
        return initial + self.expert(residual_input)


def skeleton_recall_loss(mask_logits: torch.Tensor, centerline: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = mask_logits.sigmoid()
    dims = tuple(range(1, probability.ndim))
    recall = (probability * centerline).sum(dims) / (centerline.sum(dims) + eps)
    valid = centerline.sum(dims) > 0
    return (1.0 - recall[valid]).mean() if valid.any() else probability.new_zeros(())


def _head_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def semantic_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    centerline_weight: float,
    boundary_weight: float,
    skeleton_recall_weight: float,
    cldice_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_logits, centerline_logits, boundary_logits = logits[:, 0:1], logits[:, 1:2], logits[:, 2:3]
    mask, centerline, boundary = target[:, 0:1], target[:, 1:2], target[:, 2:3]
    region = _head_loss(mask_logits, mask)
    center = _head_loss(centerline_logits, centerline) if centerline_weight > 0 else logits.new_zeros(())
    edge = _head_loss(boundary_logits, boundary) if boundary_weight > 0 else logits.new_zeros(())
    skeleton_recall = skeleton_recall_loss(mask_logits, centerline) if skeleton_recall_weight > 0 else logits.new_zeros(())
    topology = cldice_loss(mask_logits, mask) if cldice_weight > 0 else logits.new_zeros(())
    total = region + centerline_weight * center + boundary_weight * edge + skeleton_recall_weight * skeleton_recall + cldice_weight * topology
    values = {
        "region": float(region.detach()),
        "centerline": float(center.detach()),
        "boundary": float(edge.detach()),
        "skeleton_recall": float(skeleton_recall.detach()),
        "cldice": float(topology.detach()),
    }
    return total, values

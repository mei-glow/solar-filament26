from __future__ import annotations

import torch
import torch.nn.functional as F


def build_refiner(architecture: str, encoder: str, encoder_weights: str | None = "imagenet", in_channels: int = 3):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise RuntimeError("Install segmentation-models-pytorch from requirements.txt") from error
    if architecture == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=1)
    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=1)
    raise ValueError(f"Unknown refiner architecture: {architecture}")


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = logits.sigmoid()
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def _soft_erode(image: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-image, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-image, (1, 3), (1, 1), (0, 1))
    return torch.minimum(p1, p2)


def _soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, 3, 1, 1)


def _soft_skeleton(image: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    opened = _soft_dilate(_soft_erode(image))
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = _soft_erode(image)
        opened = _soft_dilate(_soft_erode(image))
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = logits.sigmoid()
    predicted_skeleton = _soft_skeleton(probability)
    target_skeleton = _soft_skeleton(target)
    dims = tuple(range(1, probability.ndim))
    precision = (predicted_skeleton * target).sum(dims) / (predicted_skeleton.sum(dims) + eps)
    recall = (target_skeleton * probability).sum(dims) / (target_skeleton.sum(dims) + eps)
    return (1.0 - (2.0 * precision * recall + eps) / (precision + recall + eps)).mean()


def refiner_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float = 0.0,
    skeleton_weight: float = 0.0,
    sample_weight: torch.Tensor | None = None,
    exclusion: torch.Tensor | None = None,
    exclusion_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    boundary = F.max_pool2d(target, 5, 1, 2) - (-F.max_pool2d(-target, 5, 1, 2))
    pixel_weight = 1.0 + boundary_weight * boundary
    bce_map = F.binary_cross_entropy_with_logits(logits, target, weight=pixel_weight, reduction="none")
    bce_per_sample = bce_map.flatten(1).mean(1)
    probability = logits.sigmoid()
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    dice_per_sample = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    if sample_weight is None:
        sample_weight = torch.ones_like(bce_per_sample)
    sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
    bce = (bce_per_sample * sample_weight).mean()
    dice = (dice_per_sample * sample_weight).mean()
    skeleton = cldice_loss(logits, target) if skeleton_weight > 0 else logits.new_zeros(())
    exclusion_loss = logits.new_zeros(())
    if exclusion is not None and exclusion_weight > 0:
        per_sample = (probability * exclusion).flatten(1).sum(1) / exclusion.flatten(1).sum(1).clamp_min(1.0)
        exclusion_loss = (per_sample * sample_weight).mean()
    total = bce + dice + skeleton_weight * skeleton + exclusion_weight * exclusion_loss
    return total, {"bce": float(bce.detach()), "dice_loss": float(dice.detach()), "skeleton_loss": float(skeleton.detach()),
                   "exclusion_loss": float(exclusion_loss.detach())}

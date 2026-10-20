"""Supervised contrastive learning on animal-level training representations."""

import math

import torch
import torch.nn.functional as F


def validate_supcon_options(args):
    """Validate contrastive settings and model compatibility."""
    weight = getattr(args, "supcon_weight", 0.0)
    temperature = getattr(args, "supcon_temperature", 0.1)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("--supcon_weight must be finite and nonnegative.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("--supcon_temperature must be finite and positive.")
    if weight == 0:
        return
    if getattr(args, "shots", 2) < 2:
        raise ValueError("Supervised contrastive training needs at least 2 training animals per class.")
    if getattr(args, "augment", False):
        raise ValueError("Contrastive experiments currently use original images; omit --augment.")
    supported = {"abmil_adapter", "zsmil", "taskres", "dual_adapt", "nl_taskres", "affine_taskres"}
    if args.model not in supported and not (
        args.model == "max_pool" and getattr(args, "max_pool_head", "none") == "adapter"
    ):
        raise ValueError("Contrastive training needs a trainable visual representation (MIL or a visual adapter).")


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """One embedding per animal; same-class *other animals* are positives.

    Every anchor must have a same-class partner and an opposite-class animal.
    All non-self animals appear in the denominator. No patch or held-out labels
    are used, and there is no projection network or additional trainable weight.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be finite and positive.")
    labels = labels.reshape(-1).to(embeddings.device)
    if embeddings.ndim != 2 or len(embeddings) != len(labels) or len(labels) < 4:
        raise ValueError("Contrastive loss expects at least four animal embeddings and matching labels.")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Contrastive embeddings must be finite.")
    same_class = labels[:, None].eq(labels[None, :])
    self_mask = torch.eye(len(labels), device=embeddings.device, dtype=torch.bool)
    positives = same_class & ~self_mask
    positive_counts = positives.sum(dim=1)
    if (positive_counts == 0).any() or same_class.all(dim=1).any():
        raise ValueError("Each animal needs another same-class animal and an opposite-class animal.")
    # Compute similarities/log-softmax in float32, including under CUDA AMP.
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        normalized = F.normalize(embeddings.float(), dim=1)
        logits = (normalized @ normalized.t()) / temperature
        logits = logits.masked_fill(self_mask, -torch.inf)
        log_probabilities = F.log_softmax(logits, dim=1)
        positive_log_probabilities = log_probabilities.masked_fill(~positives, 0)
        return -(positive_log_probabilities.sum(dim=1) / positive_counts).mean()

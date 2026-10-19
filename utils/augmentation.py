"""Reproducible image augmentation and separately cached frozen CONCH views."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .conch_features import FEATURE_VERSION, sha256_file


AUGMENTATION_VERSION = "rot90_flips_he_od_v1"


class HistologyAugment:
    """Right-angle rotations, flips, and ±10% H&E concentration changes.

    The RGB stain directions are fixed, not estimated from any study subjects.
    Each patch/view has its own seed, independent of worker scheduling and of
    model training RNG. Background and non-H&E optical-density residuals persist.
    """

    def __init__(self, seed=42, subject_id="", view=0, stain_strength=0.10):
        if not 0 <= stain_strength <= 0.25:
            raise ValueError("stain_strength must be between 0 and 0.25")
        self.seed, self.subject_id, self.view = seed, str(subject_id), view
        self.stain_strength = stain_strength
        stains = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]], dtype=np.float32)
        self.stains = stains / np.linalg.norm(stains, axis=1, keepdims=True)
        self.inverse_stains = np.linalg.pinv(self.stains)

    def __call__(self, image, patch_index):
        key = f"{self.seed}:{self.subject_id}:{self.view}:{patch_index}".encode()
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        pixels = np.asarray(image.convert("RGB"))
        pixels = np.rot90(pixels, k=int(rng.integers(4)))
        if rng.random() < 0.5:
            pixels = np.flip(pixels, axis=1)
        if rng.random() < 0.5:
            pixels = np.flip(pixels, axis=0)
        od = -np.log((pixels.astype(np.float32) + 1.0) / 256.0)
        concentration = od @ self.inverse_stains
        residual = od - concentration @ self.stains
        factors = rng.uniform(1 - self.stain_strength, 1 + self.stain_strength, size=2)
        changed_od = (concentration * factors) @ self.stains + residual
        changed_rgb = np.exp(-np.maximum(changed_od, 0)) * 256.0 - 1.0
        return Image.fromarray(np.clip(np.rint(changed_rgb), 0, 255).astype(np.uint8))


def load_augmented_views(path, original, expected_subject=None):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if expected_subject is not None and data.get("subject_id") != expected_subject:
        raise ValueError(f"Augmentation subject mismatch: {path}")
    views = data.get("augmented_features", [])
    if not views:
        raise ValueError(f"No augmented views in {path}")
    for view in views:
        if view.shape != original.shape or not torch.isfinite(view).all():
            raise ValueError(f"Invalid augmented feature bag in {path}")
    return views


def extract_augmented_features(device, subject_ids, features_dir, patches_dir,
                               raw_wsis_dir, checkpoint_path, output_dir,
                               num_views=2, seed=42, batch_size=32):
    """Generate only requested training-subject views; never replace raw bags."""
    from .train_eval import WSIPatchDataset

    if num_views < 1:
        raise ValueError("num_views must be positive")
    features_dir, output_dir = Path(features_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "augmentation_version": AUGMENTATION_VERSION,
        "feature_version": FEATURE_VERSION,
        "raw_manifest_sha256": sha256_file(features_dir / "extraction_manifest.json"),
        "implementation_sha256": sha256_file(__file__),
        "num_views": num_views, "seed": seed, "stain_strength": 0.10,
    }
    manifest = output_dir / "augmentation_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != contract:
            raise ValueError("Augmentation cache settings changed; use a separate cache directory.")
    elif any(output_dir.glob("*_aug_features.pt")):
        raise ValueError("Augmented features have no manifest; refusing unverified cache.")
    else:
        manifest.write_text(json.dumps(contract, indent=2) + "\n")
    missing = [sid for sid in sorted(subject_ids) if not (output_dir / f"{sid}_aug_features.pt").exists()]
    if not missing:
        return
    from conch.open_clip_custom import create_model_from_pretrained
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", str(checkpoint_path))
    model = model.to(device).eval()
    raw_manifest = json.loads((features_dir / "extraction_manifest.json").read_text())
    patch_size, level = raw_manifest["patch_size"], raw_manifest["level"]
    for sid in tqdm(missing, desc="Extracting augmented TRAINING views"):
        patch_data = torch.load(Path(patches_dir) / f"{sid}_patches.pt", map_location="cpu", weights_only=False)
        if patch_data["subject_id"] != sid:
            raise ValueError(f"Coordinate subject mismatch for {sid}")
        coords = patch_data["coords"]
        if len(coords) < 1:
            raise ValueError(f"No patches for {sid}")
        views = []
        for view_index in range(num_views):
            dataset = WSIPatchDataset(
                Path(raw_wsis_dir) / patch_data["slide_name"], coords,
                patch_size=patch_size, level=level, preprocess=preprocess,
                augmentation=HistologyAugment(seed, sid, view_index),
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4 if device.type == "cuda" else 0,
                                pin_memory=device.type == "cuda")
            chunks = []
            try:
                with torch.inference_mode():
                    for images in loader:
                        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                            encoded = model.encode_image(images.to(device), proj_contrast=True, normalize=True)
                        # Match the raw extractor's normalization/precision order.
                        chunks.append(F.normalize(encoded, dim=-1).cpu().float())
            finally:
                if dataset.slide is not None:
                    dataset.slide.close()
            views.append(torch.cat(chunks))
        destination = output_dir / f"{sid}_aug_features.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save({"subject_id": sid, "coords": coords, "augmented_features": views,
                    "augmentation_version": AUGMENTATION_VERSION}, temporary)
        temporary.replace(destination)

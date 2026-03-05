#prepare_dataset.py
"""
Dataset Preparation Script: Extracts non-saturated tissue patches from WSIs.
"""
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm 

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Configuration ---
PATCH_SIZE = 224
LEVEL = 0              
S_MAX = 0.10          

SELECTED_SVS_FILENAMES = [
    "Coh7a_10095_Nx_Liver_HE.svs", "Coh7a_10093_Nx_Liver_HE.svs",
    "Coh7a_10094_Nx_Liver_HE.svs", "Coh7a_10096_Nx_Liver_HE.svs",
    "Coh7a_10088_Nx_Liver_HE.svs", "Coh7a_10097_Nx_Liver_HE.svs",
    "Coh7a_10091_Nx_Liver_HE.svs", "Coh7a_10098_Nx_Liver_HE.svs",
]

SUBJECT_TABLE = {
    "10095": {"cohort": "Rhesus 7a", "sex": "M", "category": "ND",  "split": "train"},
    "10091": {"cohort": "Rhesus 7a", "sex": "M", "category": "VHD", "split": "train"},
    "10096": {"cohort": "Rhesus 7a", "sex": "M", "category": "ND",  "split": "val"},
    "10098": {"cohort": "Rhesus 7a", "sex": "M", "category": "VHD", "split": "val"},
    "10093": {"cohort": "Rhesus 7a", "sex": "M", "category": "ND",  "split": "test"},
    "10094": {"cohort": "Rhesus 7a", "sex": "M", "category": "ND",  "split": "test"},
    "10088": {"cohort": "Rhesus 7a", "sex": "M", "category": "HD",  "split": "test"},
    "10097": {"cohort": "Rhesus 7a", "sex": "M", "category": "HD",  "split": "test"},
}

@dataclass
class PrepPaths:
    project_root: Path
    dataset_dir: Path
    prepared_dir: Path

# --- Utility Functions ---
def slide_stem_safe(stem: str) -> str:
    """Sanitizes filename strings."""
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", stem)

def parse_subject_id(filename: str) -> str:
    """Extracts subject ID using regex."""
    m = re.search(r"Coh7a_(\d+)_", filename)
    return m.group(1) if m else "UNKNOWN"

def discover_selected_slides(dataset_dir: Path) -> Tuple[List[Path], List[str]]:
    """Identifies which slides are present locally vs missing."""
    all_map = {p.name: p for p in dataset_dir.glob("*.svs")}
    selected_paths, missing = [], []
    for name in SELECTED_SVS_FILENAMES:
        if name in all_map: 
            selected_paths.append(all_map[name])
        else: 
            missing.append(name)
    return selected_paths, missing

def low_saturation_fraction(patch_rgb_uint8: np.ndarray, s_max: float) -> float:
    """Calculates fraction of pixels below the saturation threshold."""
    rgb = patch_rgb_uint8.astype(np.float32) / 255.0
    V, m = rgb.max(axis=-1), rgb.min(axis=-1)
    delta = V - m
    S = np.zeros_like(V, dtype=np.float32)
    nz = V > 1e-8
    S[nz] = delta[nz] / V[nz]
    return float((S < s_max).mean())

def patch_coords_for_level(slide: openslide.OpenSlide, level: int, patch_size: int):
    """Generator for tile coordinates at a specific resolution level."""
    downsample = float(slide.level_downsamples[level])
    Wl, Hl = slide.level_dimensions[level]
    for j in range(Hl // patch_size):
        y0 = int(j * patch_size * downsample)
        for i in range(Wl // patch_size):
            x0 = int(i * patch_size * downsample)
            yield (x0, y0)

# --- Visualization & Mapping ---
def generate_maps_from_index(project_root: Path):
    """Draws bounding boxes over thumbnail maps based on extraction coordinates."""
    logging.info("Generating Patch Maps & Coverage Stats...")
    prepared_dir = project_root / "dataset" / "prepared_dataset"
    wsi_dir = project_root / "dataset" / "raw_wsis"
    index_path = prepared_dir / "index.csv"
    maps_dir = project_root / "outputs" / "patching_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    
    if not index_path.exists(): 
        logging.warning("Index path missing, aborting maps.")
        return
        
    df = pd.read_csv(index_path)
    slides = df['slide_name'].unique()
    scale_factor, patch_size = 32, 224
    patch_size_thumb = max(1, patch_size // scale_factor)

    for slide_name in tqdm(slides, desc="Processing Visuals"):
        wsi_path = wsi_dir / slide_name
        if not wsi_path.exists(): 
            continue
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            Wl, Hl = slide.level_dimensions[0]
            total_grid_patches = (Wl // patch_size) * (Hl // patch_size)
            
            thumb = slide.get_thumbnail((Wl // scale_factor, Hl // scale_factor)).convert("RGBA")
            mask = Image.new("L", thumb.size, 200) 
            draw_mask = ImageDraw.Draw(mask)
            
            slide_df = df[df['slide_name'] == slide_name]
            kept_count = len(slide_df)
            
            for _, row in slide_df.iterrows():
                tx, ty = int(row['x0'] // scale_factor), int(row['y0'] // scale_factor)
                draw_mask.rectangle([tx, ty, tx + patch_size_thumb, ty + patch_size_thumb], fill=0)
            
            overlay = Image.new("RGB", thumb.size, (0, 0, 0))
            overlay.putalpha(mask)
            Image.alpha_composite(thumb, overlay).save(maps_dir / f"{slide_name}_patch_map.png")
            
            coverage = (kept_count / total_grid_patches) * 100
            logging.info(f"Slide: {slide_name:<30} | Kept: {kept_count}/{total_grid_patches} patches ({coverage:.2f}% tissue)")
            slide.close()
        except Exception as e: 
            logging.error(f"Error drawing map for {slide_name}: {e}")

# --- Core Processing Worker ---
def _process_one_slide_worker(slide_path_str: str, prepared_dir_str: str, patch_size: int, level: int, s_max: float, subject_table: Dict) -> Tuple[str, List[Dict]]:
    """Multiprocessing worker function to extract tiles for a single WSI."""
    slide_path, prepared_dir = Path(slide_path_str), Path(prepared_dir_str)
    slide_name = slide_path.name
    subject_id = parse_subject_id(slide_name)
    if subject_id not in subject_table: 
        return slide_name, []

    meta = subject_table[subject_id]
    slide = openslide.OpenSlide(str(slide_path))
    Wl, Hl = slide.level_dimensions[level]
    total_patches = (Wl // patch_size) * (Hl // patch_size)
    
    if total_patches == 0: 
        slide.close()
        return slide_name, []

    coords, Ds = [], []
    for (x0, y0) in patch_coords_for_level(slide, level, patch_size):
        patch = np.array(slide.read_region((x0, y0), level, (patch_size, patch_size)).convert("RGB"))
        Ds.append(low_saturation_fraction(patch, s_max))
        coords.append((x0, y0))

    threshold = 0.15
    kept_indices = [i for i, d in enumerate(Ds) if d <= threshold]
    if not kept_indices: 
        slide.close()
        return slide_name, []

    patches_tensor = torch.zeros((len(kept_indices), 3, patch_size, patch_size), dtype=torch.uint8)
    coords_tensor = torch.zeros((len(kept_indices), 2), dtype=torch.int32)
    index_rows = []

    for out_i, idx in enumerate(kept_indices):
        x0, y0 = coords[idx]
        patch = np.array(slide.read_region((x0, y0), level, (patch_size, patch_size)).convert("RGB"))
        patches_tensor[out_i] = torch.from_numpy(patch).permute(2, 0, 1)
        coords_tensor[out_i] = torch.tensor([x0, y0])
        index_rows.append({"split": meta["split"], "category": meta["category"], "slide_name": slide_name, "x0": x0, "y0": y0})

    torch.save({"patches": patches_tensor, "coords": coords_tensor, "slide_name": slide_name}, 
               prepared_dir / meta["split"] / f"{subject_id}_{slide_stem_safe(slide_path.stem)}.pt")
    slide.close()
    return slide_name, index_rows

# --- Main Preparation Logic ---
def prepare_dataset(project_root: Path, overwrite: bool = False):
    """Main pipeline execution for dataset patching."""
    dataset_dir = project_root / "dataset"
    prepared_dir = dataset_dir / "prepared_dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    
    if (not overwrite) and any(prepared_dir.iterdir()): 
        logging.info("Dataset already prepared and overwrite is False. Exiting.")
        return
        
    for split in ["train", "val", "test"]: 
        (prepared_dir / split).mkdir(parents=True, exist_ok=True)
        
    selected_paths, _ = discover_selected_slides(dataset_dir / "raw_wsis")
    
    all_rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_process_one_slide_worker, str(sp), str(prepared_dir), PATCH_SIZE, LEVEL, S_MAX, SUBJECT_TABLE) for sp in selected_paths]
        for fut in tqdm(as_completed(futures), total=len(selected_paths), desc="Extracting Patches"):
            _, rows = fut.result()
            all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(prepared_dir / "index.csv", index=False)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    prepare_dataset(root, overwrite=True)
    generate_maps_from_index(root)

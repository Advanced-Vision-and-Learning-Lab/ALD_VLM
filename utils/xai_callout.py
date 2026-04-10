"""
xai_callout.py

Module for generating interpretability visualizations for Whole Slide Images (WSIs).
Produces a side-by-side figure containing a global spatial attention heatmap 
and a high-resolution regional callout centered on the densest attention cluster.
Utilizes the trained model's adapted embedding space to match the highlighted 
region with aligned semantic text concepts.
"""

import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import openslide
import textwrap

def generate_zoomed_callout(
    wsi_id,
    slide_path,
    coords,
    importance_scores,
    features,
    concept_matrix,
    concept_phrases,
    patch_size,
    level,
    out_dir,
    model,        
    device,     
    zoom_window_patches=6
):
    """
    Locates the highest-density attention region, extracts the corresponding image crop,
    computes the region's adapted semantic embedding via the trained model, and outputs
    a visualized bounding-box callout aligned with relevant VLM text concepts.

    Args:
        wsi_id (str): Unique identifier for the Whole Slide Image.
        slide_path (str/Path): File path to the raw .svs or .ndpi slide.
        coords (np.ndarray): Array of spatial coordinates (x, y) for the extracted patches.
        importance_scores (np.ndarray): 1D array of attention/importance weights per patch.
        features (torch.Tensor): Pre-extracted visual features for the patches.
        concept_matrix (np.ndarray): Matrix of embedded text concepts.
        concept_phrases (list): List of text strings corresponding to the concept_matrix.
        patch_size (int): Pixel dimension of the patches (e.g., 224).
        level (int): Slide magnification level used for extraction.
        out_dir (Path): Output directory for the generated visualizations.
        model (nn.Module): The trained MIL/Adapter model for regional feature evaluation.
        device (torch.device): Compute device (CPU/GPU).
        zoom_window_patches (int): Defines the dimension (N x N patches) of the callout bounding box.
    """
    callout_dir = out_dir / "zoomed_callouts"
    callout_dir.mkdir(parents=True, exist_ok=True)
    
    slide = openslide.OpenSlide(str(slide_path))
    thumb_factor = 32
    Wl, Hl = slide.level_dimensions[level]
    thumb = np.array(slide.get_thumbnail((Wl // thumb_factor, Hl // thumb_factor)).convert("RGB"))

    # Reconstruct the spatial attention map and apply Gaussian smoothing 
    # to identify contiguous regions of high model attention.
    heatmap_grid = np.zeros((Hl // thumb_factor, Wl // thumb_factor), dtype=np.float32)
    tissue_mask = np.zeros((Hl // thumb_factor, Wl // thumb_factor), dtype=bool)
    p_size_thumb = patch_size // thumb_factor

    for idx, (x, y) in enumerate(coords):
        tx, ty = int(x // thumb_factor), int(y // thumb_factor)
        heatmap_grid[ty:ty+p_size_thumb, tx:tx+p_size_thumb] = importance_scores[idx]
        tissue_mask[ty:ty+p_size_thumb, tx:tx+p_size_thumb] = True

    heatmap_smoothed = scipy.ndimage.gaussian_filter(heatmap_grid, sigma=p_size_thumb)
    
    if tissue_mask.any():
        tissue_min = heatmap_smoothed[tissue_mask].min()
        tissue_max = heatmap_smoothed[tissue_mask].max()
    else:
        tissue_min, tissue_max = 0.0, 1.0

    heatmap_norm = (heatmap_smoothed - tissue_min) / (tissue_max - tissue_min + 1e-8)
    heatmap_norm = np.clip(heatmap_norm, 0.0, 1.0)

    cmap = plt.cm.jet
    overlay = cmap(heatmap_norm)
    overlay[..., 3] = np.where(tissue_mask, heatmap_norm * 0.6, 0.0)

    # Identify the global maximum of the smoothed attention map to serve as the callout center.
    max_ty, max_tx = np.unravel_index(np.argmax(heatmap_smoothed), heatmap_smoothed.shape)
    center_x = max_tx * thumb_factor
    center_y = max_ty * thumb_factor

    # Compute bounding box coordinates for the high-resolution crop, 
    # ensuring boundaries remain strictly within the slide dimensions.
    offset = (zoom_window_patches // 2) * patch_size
    zx_start = max(0, center_x - offset)
    zy_start = max(0, center_y - offset)
    zx_end = min(Wl, center_x + offset)
    zy_end = min(Hl, center_y + offset)
    
    read_w = zx_end - zx_start
    read_h = zy_end - zy_start

    # Extract the high-resolution image crop for the callout.
    zoomed_img = slide.read_region((int(zx_start), int(zy_start)), level, (int(read_w), int(read_h))).convert("RGB")

    # Isolate the pre-extracted visual features corresponding to the coordinates within the bounding box.
    region_mask = (
        (coords[:, 0] >= zx_start) & (coords[:, 0] < zx_end) &
        (coords[:, 1] >= zy_start) & (coords[:, 1] < zy_end)
    )
    
    # Fallback heuristic: If region alignment fails due to edge truncation, default to the highest-scoring individual patch.
    if not np.any(region_mask):
        region_features = features[np.argmax(importance_scores)].unsqueeze(0)
    else:
        region_features = features[region_mask]

    region_features = region_features.to(device)

    # Perform a forward pass through the trained model to obtain the region's adapted semantic embedding.
    with torch.no_grad(), torch.autocast(device_type=device.type):
        outputs = model(region_features)
        region_emb = outputs[1] 

    # L2-normalize the adapted output representation.
    region_emb_norm = region_emb.detach().cpu().float().numpy().squeeze()

    # Perform context-aware concept filtering: restrict the semantic search space 
    # to target-class concepts to ensure clinically relevant text alignment.
    # Updated to match the vocabulary of the new bag-level ALD prompts.
    ald_keywords = [
        "steatohepatitis", 
        "chronic liver disease", 
        "macrovesicular steatosis", 
        "alcoholic liver injury", 
        "lipid accumulation"
    ]
    similarities = region_emb_norm @ concept_matrix.T
    
    valid_indices = [
        i for i, phrase in enumerate(concept_phrases) 
        if any(kw in phrase for kw in ald_keywords)
    ]
    
    # Safeguard against empty keyword matches by reverting to the full concept space.
    if not valid_indices:
        valid_indices = list(range(len(concept_phrases)))

    # Mask similarities of irrelevant concepts to prioritize target morphology.
    masked_similarities = np.full_like(similarities, -1.0)
    masked_similarities[valid_indices] = similarities[valid_indices]

    best_concept_idx = int(np.argmax(masked_similarities))
    best_phrase = concept_phrases[best_concept_idx]
    
    # Construct the side-by-side visualization figure.
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=300, gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Left Panel: Global WSI Heatmap.
    ax_main = axes[0]
    ax_main.imshow(thumb)
    ax_main.imshow(overlay)
    
    # Render a dual-layered, high-contrast bounding box on the thumbnail to clearly demarcate the callout origin.
    rect_x, rect_y = zx_start // thumb_factor, zy_start // thumb_factor
    rect_w, rect_h = read_w // thumb_factor, read_h // thumb_factor
    
    rect_bg = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=4, edgecolor='black', facecolor='none')
    rect_fg = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    
    ax_main.add_patch(rect_bg)
    ax_main.add_patch(rect_fg)
    ax_main.set_title("Whole Slide Attention Heatmap", fontsize=16, fontweight='bold')
    ax_main.axis('off')

    # Right Panel: Zoomed Regional Callout.
    ax_zoom = axes[1]
    ax_zoom.imshow(np.array(zoomed_img))
    
    # Apply a distinct border to visually separate the callout region.
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor('cyan')
        spine.set_linewidth(4)
        
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

    # Overlay the semantically matched VLM text representation.
    wrapped_text = textwrap.fill(best_phrase, width=60)
    title_text = f"High Attention Region \n\nTop Match:\n\"{wrapped_text}\""
    ax_zoom.set_title(title_text, fontsize=15, color='darkred', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(callout_dir / f"xai_callout_{wsi_id}.png", bbox_inches='tight')
    plt.close()
    slide.close()

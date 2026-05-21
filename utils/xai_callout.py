"""
xai_callout.py
Module for generating interpretability visualizations for Whole Slide Images (WSIs).
"""

import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import openslide

def generate_zoomed_callout(
    wsi_id,
    slide_path,
    coords,
    importance_scores,
    patch_size,
    level,
    out_dir,
    local_concept_text=None,
    zoom_window_patches=6
):
    """
    Locates the highest-density attention region, extracts the corresponding image crop,
    and outputs a visualized bounding-box callout.
    """
    callout_dir = out_dir / "zoomed_callouts"
    callout_dir.mkdir(parents=True, exist_ok=True)
    
    slide = openslide.OpenSlide(str(slide_path))
    thumb_factor = 32
    
    Wl, Hl = slide.level_dimensions[level]

    thumb_arr = np.array(slide.get_thumbnail((Wl // thumb_factor, Hl // thumb_factor)).convert("RGB"))
    half_w_thumb = (Wl // 2) // thumb_factor
    thumb = thumb_arr[:, :half_w_thumb]
    
    Wl = Wl // 2

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

    max_ty, max_tx = np.unravel_index(np.argmax(heatmap_smoothed), heatmap_smoothed.shape)
    center_x = max_tx * thumb_factor
    center_y = max_ty * thumb_factor

    offset = (zoom_window_patches // 2) * patch_size
    zx_start = max(0, center_x - offset)
    zy_start = max(0, center_y - offset)
    zx_end = min(Wl, center_x + offset)
    zy_end = min(Hl, center_y + offset)
    
    read_w = zx_end - zx_start
    read_h = zy_end - zy_start

    zoomed_img = slide.read_region((int(zx_start), int(zy_start)), level, (int(read_w), int(read_h))).convert("RGB")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=300, gridspec_kw={'width_ratios': [1, 1.2]})
    
    ax_main = axes[0]
    ax_main.imshow(thumb)
    ax_main.imshow(overlay)
    
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
    
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor('cyan')
        spine.set_linewidth(4)
        
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

    title_text = "High Attention Region"
    if local_concept_text:
        title_text += f"\n\nLocal Concept Match:\n{local_concept_text}"
        
    ax_zoom.set_title(title_text, fontsize=14, color='darkred', fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(callout_dir / f"xai_callout_{wsi_id}.png", bbox_inches='tight')
    plt.close()
    slide.close()

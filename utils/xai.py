"""
xai.py
Explainable AI (xAI) Module.
"""

import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openslide
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .mil_models import MaxPoolModel, build_model

from .xai_callout import generate_zoomed_callout 

def _initialize_xai_model(args, bag_embeddings, instance_embeddings, device):
    """
    Reconstructs the specified model architecture for xAI evaluation.
    """
    return build_model(args, bag_embeddings).to(device)


def concept_embeddings(model, concept_matrix):
    """Prompt retrieval is descriptive; mean-mode residuals are class-level only."""
    text = torch.as_tensor(concept_matrix, device=model.text_embeddings.device)
    # Mean models learn two prototypes, not 20 independent prompt residuals.
    # Retrieve against frozen prompts in that case, with an explicit report label.
    if model.text_pool == "mean":
        return F.normalize(text, p=2, dim=-1)
    if hasattr(model, "task_scale"):
        text = model.task_scale * text + model.alpha * model.task_shift
    elif hasattr(model, "task_res"):
        text = text + model.alpha * model.task_res
    elif hasattr(model, "semantic_adapter"):
        text = text + model.alpha * model.semantic_adapter(text)
    return F.normalize(text, p=2, dim=-1)


def compute_integrated_gradients(model, features, device, steps=50):
    """
    Computes Integrated Gradients for baseline models lacking attention mechanisms.
    """
    if isinstance(model, MaxPoolModel):
        raise ValueError("Zero-reference integrated gradients are invalid for this normalized Max Pool path; validate a reference before generating attributions.")
    baseline = torch.zeros_like(features)
    avg_grads = torch.zeros_like(features)
    
    for i in range(1, steps + 1):
        alpha = i / steps
        step_feat = (baseline + alpha * (features - baseline)).detach().requires_grad_(True)
        
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(step_feat)
            logits = outputs[0]
            target_score = logits[0, 1] 
        
        model.zero_grad()
        target_score.backward()
        avg_grads += step_feat.grad / steps
        
    ig_attributions = (features - baseline) * avg_grads
    importance_scores = torch.linalg.norm(ig_attributions, dim=1).cpu().numpy()
    
    return importance_scores

def run_xai_analysis(xai_infos, args, bag_embeddings, instance_embeddings, device, features_dir, raw_wsis_map, results_dir, patch_size, level=0):
    """
    Executes the full xAI pipeline. Extracts highly weighted patches and regions,
    passes them through the trained model.
    """
    if not xai_infos:
        return

    if args.model == "max_pool":
        logging.warning("Skipping Max Pool xAI: zero-reference integrated gradients failed the completeness diagnostic.")
        return
    else:
        xai_mode = "Visual-Attention"
    
    # Load the exact text embeddings used during training
    train_text_dict = torch.load(features_dir / "training_text_embeddings.pt", map_location="cpu", weights_only=False)
    concept_phrases = list(train_text_dict.keys())
    concept_matrix = np.stack(list(train_text_dict.values()))
    
    target_sentences = concept_phrases

    for xai_info in xai_infos:
        wsi_id = xai_info['wsi_id']
        iter_idx = xai_info['combo_idx']
        seed_idx = xai_info['seed_idx']
        
        out_dir = results_dir / f"xAI_Iter{iter_idx:02d}_Seed{seed_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Running {xai_mode} xAI Analysis on WSI {wsi_id}... Saving to {out_dir.name}")
        
        model = _initialize_xai_model(
            args=args, 
            bag_embeddings=bag_embeddings,
            instance_embeddings=instance_embeddings, 
            device=device
        )
        
        model.load_state_dict({k: v.to(device) for k, v in xai_info["weights"].items()})
        model.eval()

        feat_data = torch.load(features_dir / f"{wsi_id}_features.pt", map_location="cpu", weights_only=False)
        coords = feat_data["coords"].numpy()
        features = feat_data["features"].to(device)
        
        mass_msg = ""
        
        # --- Importance Score Extraction ---
        if xai_mode == "Visual-Attention":
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(features)
                slide_emb = outputs[1]
                w = outputs[2]
                importance_scores = w.reshape(-1).float().cpu().numpy()
            
            sorted_scores = np.sort(importance_scores)[::-1]
            cum_scores = np.cumsum(sorted_scores)
            patches_50 = int(np.argmax(cum_scores >= 0.5) + 1) 
            total_p = len(importance_scores)
            pct_of_tissue = (patches_50 / total_p) * 100
            
            mass_msg = (f"\n--- Attention Mass Diagnostics ---\n"
                        f"To reach 50% of the total attention weight, the attention was concentrated on:\n"
                        f"{patches_50} out of {total_p} total patches ({pct_of_tissue:.2f}% of the tissue).\n")
        else:
            importance_scores = compute_integrated_gradients(model, features, device)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(features)
                slide_emb = outputs[1]
            mass_msg = "\n--- Attention Mass Diagnostics ---\nN/A for Gradient-Based methods.\n"
            
        # --- Slide-Level Text Semantic Matching ---
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            all_text_norm = concept_embeddings(model, concept_matrix)
            slide_emb_norm = F.normalize(slide_emb, p=2, dim=-1)
            
            # Compute similarities against shifted prompts
            text_sims = (slide_emb_norm @ all_text_norm.t()).squeeze()
            
            top_k = 3
            top_scores, top_indices = torch.topk(text_sims, k=top_k)
            
            text_msg = ("\nPrompt retrieval is descriptive similarity, not lesion detection or causal attribution.\n"
                        + ("Mean-mode retrieval uses frozen individual prompts; learned residuals belong to class prototypes.\n" if args.text_pool == "mean" else "")
                        + f"\n--- Top {top_k} Matched Semantic Concepts ---\n")
            for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
                text_msg += (f"Rank {rank + 1}:\n"
                             f"  Concept: '{target_sentences[idx]}'\n"
                             f"  Similarity: {score:.4f}\n")
            
        # ==========================================
        # Heatmap Generation
        # ==========================================
        slide_path = raw_wsis_map[wsi_id]
        slide = openslide.OpenSlide(str(slide_path))
        thumb_factor = 32
        
        Wl, Hl = slide.level_dimensions[level]
        
        full_thumb = slide.get_thumbnail((Wl // thumb_factor, Hl // thumb_factor))
        thumb_arr = np.array(full_thumb.convert("RGB"))
        half_w_thumb = (Wl // 2) // thumb_factor
        thumb = thumb_arr[:, :half_w_thumb] # Crop visual to left half
        
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
        
        fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
        ax.imshow(thumb)
        
        cmap = plt.cm.jet
        overlay = cmap(heatmap_norm)
        overlay[..., 3] = np.where(tissue_mask, heatmap_norm * 0.6, 0.0)
        
        ax.imshow(overlay)
        ax.set_title(f"{xai_mode} Heatmap: {wsi_id}", fontsize=16)
        ax.axis('off')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([]) 
        cbar = fig.colorbar(sm, cax=cax)
        
        cbar.set_label('Relative Patch Importance', rotation=270, labelpad=25, fontsize=16)
        
        cbar.ax.tick_params(labelsize=14)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"xai_heatmap_overlay_{wsi_id}.png", bbox_inches='tight')
        plt.close()
        
        if xai_mode == "Visual-Attention":
            local_text_match = None
            
            if args.text_pool == "lse":
                p_size_thumb = patch_size // thumb_factor
                heatmap_smoothed_flat = scipy.ndimage.gaussian_filter(heatmap_grid, sigma=p_size_thumb)
                max_ty, max_tx = np.unravel_index(np.argmax(heatmap_smoothed_flat), heatmap_smoothed_flat.shape)
                center_x = max_tx * thumb_factor
                center_y = max_ty * thumb_factor
                
                offset = (6 // 2) * patch_size # zoom_window_patches = 6
                zx_start = center_x - offset
                zy_start = center_y - offset
                zx_end = center_x + offset
                zy_end = center_y + offset
                
                local_features = []
                for idx, (x, y) in enumerate(coords):
                    if zx_start <= x <= zx_end and zy_start <= y <= zy_end:
                        local_features.append(features[idx])
                        
                if local_features:
                    # 3. Aggregate local features
                    local_feats_tensor = torch.stack(local_features)
                    
                    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        local_emb, _ = model.aggregator(local_feats_tensor)
                        
                        # Apply Adapter if present
                        if hasattr(model, 'adapter'):
                            res = model.adapter(local_emb)
                            ratio = getattr(model, 'ratio', 0.2)
                            local_emb = ratio * res + (1 - ratio) * local_emb
                            
                        local_emb_norm = F.normalize(local_emb, p=2, dim=-1)
                        
                        local_text_sims = (local_emb_norm @ all_text_norm.t()).squeeze()
                        best_local_idx = torch.argmax(local_text_sims).item()
                        best_local_score = local_text_sims[best_local_idx].item()
                        
                        local_text_match = f"'{target_sentences[best_local_idx]}'"

                        text_msg += f"\n--- Regional Callout Semantic Match ---\n{local_text_match} (Sim: {best_local_score:.2f})\n"
                        
            generate_zoomed_callout(
                wsi_id=wsi_id,
                slide_path=slide_path,
                coords=coords,
                importance_scores=importance_scores,
                patch_size=patch_size,
                level=level,
                out_dir=out_dir,
                local_concept_text=local_text_match
            )
        
        # ==========================================
        # Top-K Patch Extraction
        # ==========================================
        top_k = 15
        top_indices = np.argsort(importance_scores)[-top_k:][::-1].copy()
        fig, axes = plt.subplots(3, 5, figsize=(15, 9), dpi=300) 
        axes = axes.flatten()
        
        for i, (ax_, idx) in enumerate(zip(axes, top_indices)):
            x, y = coords[idx]
            patch_img = np.array(slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB"))
            ax_.imshow(patch_img)
            ax_.set_title(f"Rank {i+1}", fontsize=12)
            for spine in ax_.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
            ax_.set_xticks([])
            ax_.set_yticks([])
            
        plt.tight_layout()
        plt.savefig(out_dir / f"xai_top_patches_{wsi_id}.png", bbox_inches='tight')
        plt.close()
        
        # ==========================================
        # Slide-Level Reporting
        # ==========================================
        with open(out_dir / f"xai_report_{wsi_id}.txt", "w") as f:
            f.write("=====================================================\n")
            f.write(f"      xAI VISUAL ATTENTION REPORT\n")
            f.write("=====================================================\n\n")
            f.write(f"Model Architecture Used : {args.model}\n")
            f.write(f"Evaluated WSI Subject ID: {wsi_id}\n")
            f.write(f"Generated from Iteration: {iter_idx} | Seed: {seed_idx}\n")
            
            f.write(mass_msg) 
            f.write(text_msg)
            
            f.write("\n--- Patch-Level Saliency (Top 15) ---\n")
            for rank, idx in enumerate(top_indices):
                f.write(f"Rank {rank+1:02d}:\n")
                f.write(f"  Coordinates : X={coords[idx][0]:<6}, Y={coords[idx][1]}\n")
                f.write(f"  Score       : {importance_scores[idx]:.6f}\n")
            
        logging.info(f"xAI Analysis complete for WSI {wsi_id}.")
        
        slide.close()

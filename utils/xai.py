"""
xai.py
Explainable AI (xAI) Module.
Interprets MIL model decisions through semantic attention heatmaps, 
Integrated Gradients, and fine-tuned text-to-patch alignment.
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

from .mil_models import (
    ABMILAdapterModel,
    ZSMILModel,
    TaskResModel,
    MaxPoolModel,
    PromptLearningMILModel,
    TOPModel
)

def _initialize_xai_model(args, text_prototypes, device, clip_model=None, tokenizer=None, bag_concepts=None, instance_concepts=None, bag_templates=None, instance_templates=None):
    """
    Reconstructs the specified model architecture for xAI evaluation.
    """
    if args.model == "abmil_adapter":
        model = ABMILAdapterModel(text_embeddings=text_prototypes, reduction=args.reduction, abmil_d=args.abmil_d, fusion=args.fusion).to(device)
    elif args.model == "zsmil":
        model = ZSMILModel(text_embeddings=text_prototypes, abmil_d=args.abmil_d)
    elif args.model == "taskres":
        model = TaskResModel(text_embeddings=text_prototypes, alpha=0.25, abmil_d=args.abmil_d, fusion=args.fusion).to(device)
    elif args.model == "max_pool":
        model = MaxPoolModel(text_embeddings=text_prototypes, head_type=args.max_pool_head, reduction=args.reduction, alpha=0.25)
    elif args.model == "coop":
        model = PromptLearningMILModel(
            clip_model=clip_model, 
            tokenizer=tokenizer, 
            concepts_dict=instance_concepts, 
            templates=instance_templates,    
            n_ctx=args.n_ctx, 
            abmil_d=args.abmil_d,
            fusion=args.fusion,
        )
    elif args.model == "top":
        model = TOPModel(
            clip_model=clip_model, 
            tokenizer=tokenizer, 
            bag_concepts_dict=bag_concepts, 
            instance_concepts_dict=instance_concepts, 
            bag_templates=bag_templates, 
            instance_templates=instance_templates,
            n_ctx=args.n_ctx
        )    
    else:
        raise ValueError(f"Unknown model type: {args.model}")
        
    return model.to(device)


def compute_integrated_gradients(model, features, device, steps=50):
    """
    Computes Integrated Gradients for baseline models lacking explicit attention mechanisms.
    """
    baseline = torch.zeros_like(features)
    avg_grads = torch.zeros_like(features)
    
    for i in range(1, steps + 1):
        alpha = i / steps
        step_feat = (baseline + alpha * (features - baseline)).detach().requires_grad_(True)
        
        with torch.autocast(device_type=device.type):
            logits, _, _ = model(step_feat)
            target_score = logits[0, 1] 
        
        model.zero_grad()
        target_score.backward()
        avg_grads += step_feat.grad / steps
        
    ig_attributions = (features - baseline) * avg_grads
    importance_scores = torch.linalg.norm(ig_attributions, dim=1).cpu().numpy()
    
    return importance_scores


def run_xai_analysis(xai_infos, args, text_prototypes, device, features_dir, raw_wsis_map, results_dir, patch_size, level=0, clip_model=None, tokenizer=None, bag_concepts=None, instance_concepts=None, bag_templates=None, instance_templates=None):
    """
    Executes the full xAI pipeline. Extracts highly weighted patches and verifies
    their semantic alignment with fine-tuned text concepts.
    """
    if not xai_infos:
        return

    # Categorize models based on their fusion paradigm
    early_fusion_models = ["zsmil", "taskres", "coop", "top"] 
    late_fusion_models = ["abmil_adapter"]
    
    if args.model in early_fusion_models:
        xai_mode = "Semantic-Attention"
    elif args.model in late_fusion_models:
        xai_mode = "Visual-Attention"
    else:
        xai_mode = "Gradient-Based"
    
    pure_concept_dict = torch.load(features_dir / "pure_concept_embeddings.pt", map_location="cpu", weights_only=False)
    concept_phrases = list(pure_concept_dict.keys())
    concept_matrix = np.stack(list(pure_concept_dict.values()))
    
    target_concepts = [
        "macrovesicular steatosis", 
        "microvesicular lipid accumulation", 
        "hepatocyte ballooning degeneration", 
        "neutrophilic lobular inflammation", 
        "perisinusoidal fibrosis"
    ]
    concept_dict = torch.load(features_dir / "concept_embeddings.pt", map_location="cpu", weights_only=False)
    target_sentences = [sent for sent in concept_dict.keys() if any(c in sent for c in target_concepts)]

    for xai_info in xai_infos:
        wsi_id = xai_info['wsi_id']
        iter_idx = xai_info['combo_idx']
        seed_idx = xai_info['seed_idx']
        
        out_dir = results_dir / f"xAI_Iter{iter_idx:02d}_Seed{seed_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Running {xai_mode} xAI Analysis on WSI {wsi_id}... Saving to {out_dir.name}")
        
        model = _initialize_xai_model(
            args=args, 
            text_prototypes=text_prototypes, 
            device=device, 
            clip_model=clip_model, 
            tokenizer=tokenizer, 
            bag_concepts=bag_concepts, 
            instance_concepts=instance_concepts, 
            bag_templates=bag_templates, 
            instance_templates=instance_templates
        )
        
        model.load_state_dict({k: v.to(device) for k, v in xai_info["weights"].items()})
        model.eval()

        feat_data = torch.load(features_dir / f"{wsi_id}_features.pt", map_location="cpu", weights_only=False)
        coords = feat_data["coords"].numpy()
        features = feat_data["features"].to(device)
        
        mass_msg = ""
        
        # --- Importance Score Extraction ---
        if xai_mode in ["Semantic-Attention", "Visual-Attention"]:
            with torch.no_grad(), torch.autocast(device_type=device.type):
                logits, slide_emb, w = model(features)
                importance_scores = w.squeeze().cpu().numpy()
            
            sorted_scores = np.sort(importance_scores)[::-1]
            cum_scores = np.cumsum(sorted_scores)
            patches_50 = int(np.argmax(cum_scores >= 0.5) + 1) 
            total_p = len(importance_scores)
            pct_of_tissue = (patches_50 / total_p) * 100
            
            mass_msg = (f"\n--- Decision Mass Diagnostics ---\n"
                        f"To reach 50% of the total decision weight, the model relied on:\n"
                        f"{patches_50} out of {total_p} total patches ({pct_of_tissue:.2f}% of the tissue).\n")
            if xai_mode == "Semantic-Attention":
                mass_msg += "Note: In this model, these weights are directly derived from patch-to-text semantic similarity.\n"
        else:
            importance_scores = compute_integrated_gradients(model, features, device)
            with torch.no_grad(), torch.autocast(device_type=device.type):
                _, slide_emb, _ = model(features)
            mass_msg = "\n--- Decision Mass Diagnostics ---\nN/A for Gradient-Based methods.\n"
            
        # ==========================================
        # Heatmap Generation
        # ==========================================
        slide_path = raw_wsis_map[wsi_id]
        slide = openslide.OpenSlide(str(slide_path))
        thumb_factor = 32
        Wl, Hl = slide.level_dimensions[level]
        thumb = slide.get_thumbnail((Wl // thumb_factor, Hl // thumb_factor))
        
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
        cbar.set_label('Relative Patch Importance', rotation=270, labelpad=20, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"xai_heatmap_overlay_{wsi_id}.png", bbox_inches='tight')
        plt.close()
        
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
        # Concept Alignment and Reporting
        # ==========================================
        top_patch_feats = features[top_indices.copy()].detach().cpu().float()
        top_patch_feats = F.normalize(top_patch_feats, p=2, dim=-1).numpy()
        
        # --- Handle CoOp/TOP Fine-Tuned Text Space ---
        if args.model in ["coop", "top"]:
            learner = model.prompt_learner if args.model == "coop" else model.instance_learner
            active_concepts = instance_concepts if args.model == "top" else bag_concepts # bag_concepts here represents the single CONCEPTS dict passed from demo if coop
            active_templates = instance_templates if args.model == "top" else bag_templates
            
            concept_phrases = []
            for cls_name in learner.classes:
                for concept in active_concepts[cls_name]:
                    for t in active_templates:
                        concept_phrases.append(t.format(concept))
            
            with torch.no_grad(), torch.autocast(device_type=device.type):
                prompts, _ = learner()
                x = prompts + model.text_tower.positional_embedding
                x = x.permute(1, 0, 2)  
                x = model.text_tower.transformer(x, attn_mask=model.text_tower.attn_mask)
                x = x.permute(1, 0, 2)  
                x = model.text_tower.ln_final(x)
                
                eos_indices = learner.attention_mask.sum(dim=-1) - 1 + learner.n_ctx
                text_features = x[torch.arange(x.shape[0]), eos_indices]
                
                if hasattr(model.text_tower, 'text_projection') and model.text_tower.text_projection is not None:
                    text_features = text_features @ model.text_tower.text_projection
            
            concept_matrix = F.normalize(text_features, p=2, dim=-1).cpu().numpy()
            report_header = f"Top 15 Patches - Fine-Tuned {args.model.upper()} Concept Alignment"
            
        else:
            concept_matrix = np.stack(list(pure_concept_dict.values()))
            concept_phrases = list(pure_concept_dict.keys())
            report_header = "Top 15 Patches - Zero-Shot Concept Alignment"
            
        similarity_matrix = top_patch_feats @ concept_matrix.T 
        
        with open(out_dir / f"xai_top15_concepts_{wsi_id}.txt", "w") as f:
            f.write(f"{report_header}\n")
            f.write("===================================================\n")
            for i, idx in enumerate(top_indices):
                patch_sims = similarity_matrix[i]
                best_concept_idx = int(np.argmax(patch_sims))
                best_phrase = concept_phrases[best_concept_idx]
                best_score = patch_sims[best_concept_idx]
                
                f.write(f"Rank {i+1:02d} (Patch Index {idx}):\n")
                f.write(f"  Coordinates  : X={coords[idx][0]:<6}, Y={coords[idx][1]}\n")
                f.write(f"  Top Match    : {best_phrase}\n")
                f.write(f"  Cosine Score : {best_score:.4f}\n\n")
                
        # --- Slide-Level Reporting ---
        slide_emb_float = slide_emb.detach().cpu().float()
        slide_emb_norm = F.normalize(slide_emb_float, p=2, dim=-1).numpy().squeeze()
        
        sentence_scores = {}
        if args.model in ["coop", "top"]:
            for idx, phrase in enumerate(concept_phrases):
                if any(c in phrase for c in target_concepts):
                    sentence_scores[phrase] = float(np.dot(slide_emb_norm, concept_matrix[idx]))
        else:
            for sentence in target_sentences:
                sentence_scores[sentence] = float(np.dot(slide_emb_norm, concept_dict[sentence]))
                
        ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

        with open(out_dir / f"xai_report_{wsi_id}.txt", "w") as f:
            f.write("=====================================================\n")
            f.write(f"      xAI ALIGNMENT & {xai_mode.upper()} REPORT\n")
            f.write("=====================================================\n\n")
            f.write(f"Model Architecture Used : {args.model}\n")
            f.write(f"Evaluated WSI Subject ID: {wsi_id}\n")
            f.write(f"Generated from Iteration: {iter_idx} | Seed: {seed_idx}\n")
            
            f.write(mass_msg) 
            
            f.write("\n--- Slide-Level Morphological Profiling ---\n")
            f.write("Cosine Similarity (-1 to +1) of the final aggregated WSI embedding against target concepts:\n")
            for rank, (sentence, score) in enumerate(ranked_sentences):
                f.write(f"{rank+1:02d}. Score: {score:+.4f} | {sentence}\n")
            
            f.write("\n--- Patch-Level Saliency (Top 15) ---\n")
            for rank, idx in enumerate(top_indices):
                f.write(f"Rank {rank+1:02d}:\n")
                f.write(f"  Coordinates : X={coords[idx][0]:<6}, Y={coords[idx][1]}\n")
                f.write(f"  Score       : {importance_scores[idx]:.6f}\n")
            
        logging.info(f"xAI Analysis complete for WSI {wsi_id}.")
        slide.close()

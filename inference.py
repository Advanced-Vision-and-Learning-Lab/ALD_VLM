#inference.py
"""
Model evaluation, visualization generation, and logging script.
"""
import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openslide
import sklearn.metrics
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from models.models import ZSMIL, TaskRes, CLIPAdapter

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class WSIDataset(Dataset):
    """Loads feature tensors and metadata for evaluation splits."""
    def __init__(self, features_dir: str | Path, split: str = "test"):
        self.split_dir = Path(features_dir) / split
        self.pt_files = sorted(list(self.split_dir.glob("*.pt")))
        self.label_map = {"Non-Drinking": 0, "Drinking": 1}

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, torch.Tensor, str]:
        pt_path = self.pt_files[idx]
        data = torch.load(pt_path, map_location="cpu")
        subject_id = data.get("subject_id", "Unknown_Subject")
        return data["features"], self.label_map[data["binary_category"]], data["slide_name"], data["coords"], subject_id


def get_top_specific_prompt(query_feature: torch.Tensor, specific_texts_dict: dict, predicted_class: str, device: torch.device) -> tuple[str, float]:
    """Finds the specific text prompt that best matches the query feature."""
    class_data = specific_texts_dict[predicted_class]
    specific_prompts = class_data["prompts"]
    specific_features = class_data["features"].to(device) 
    
    query_norm = F.normalize(query_feature.squeeze(), p=2, dim=-1)
    similarities = query_norm @ specific_features.t()
    best_idx = similarities.argmax().item()
    
    return specific_prompts[best_idx], float(similarities[best_idx])


def analyze_visuals(slide_name: str, features: torch.Tensor, coords: torch.Tensor, model: torch.nn.Module, agg_type: str, 
                    wsi_path: Path, out_dir: Path, patch_dir: Path, device: torch.device, run_name: str, specific_texts_dict: dict):
    """Generates heatmaps, retrieves top patches, and stitches predictive clusters."""
    patch_size = 224
    scale_factor = 16
    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as e: 
        logging.warning(f"Could not open WSI {slide_name}: {e}")
        return

    model.eval()
    with torch.no_grad():
        logits, embedding, w = model(features) 
        
        if agg_type == "ABMIL":
            raw_scores = w.squeeze().cpu().numpy()
            score_type = "Attention_Weight"
        else:
            if isinstance(model, TaskRes): 
                prototype = model.text_embeddings + model.alpha * model.classifier
                p_feat = features
            elif isinstance(model, CLIPAdapter):
                res = model.adapter(features)
                prototype = model.text_embeddings
                p_feat = model.ratio * res + (1 - model.ratio) * features
            else: 
                prototype = model.text_embeddings
                p_feat = features

            p_norm, e_norm = F.normalize(prototype, p=2, dim=-1), F.normalize(p_feat, p=2, dim=-1)
            scale = model.logit_scale.exp() if model.use_learnable_scale else model.logit_scale
            
            patch_logits = (e_norm @ p_norm.t()) * scale
            raw_scores = patch_logits.softmax(dim=-1)[:, 1].cpu().numpy()
            score_type = "Cosine_Probability"
            
        if isinstance(model, CLIPAdapter):
            res = model.adapter(features)
            adapted_patch_features = model.ratio * res + (1 - model.ratio) * features
        else:
            adapted_patch_features = features
        
    s_min, s_max = raw_scores.min(), raw_scores.max()
    norm_scores = (raw_scores - s_min) / (s_max - s_min + 1e-8)

    W, H = slide.dimensions
    thumb_w, thumb_h = W // scale_factor, H // scale_factor
    
    original_thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    original_thumb.save(out_dir / f"{slide_name}_original.png", dpi=(600, 600))

    circles_img = None
    draw = None
    if agg_type == "ABMIL":
        circles_img = original_thumb.copy()
        draw = ImageDraw.Draw(circles_img)

    heatmap_array = np.full((thumb_h, thumb_w), np.nan, dtype=np.float32)
    patch_size_thumb = max(1, patch_size // scale_factor)
    coords_np = coords.numpy()
    
    for i in range(len(coords_np)):
        x0, y0 = coords_np[i]
        tx, ty = int(x0 // scale_factor), int(y0 // scale_factor)
        heatmap_array[ty:min(ty+patch_size_thumb, thumb_h), tx:min(tx+patch_size_thumb, thumb_w)] = norm_scores[i]
        
    cmap = mpl.colormaps['coolwarm'].copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))
    colored_heatmap = (cmap(heatmap_array) * 255).astype(np.uint8)
    Image.fromarray(colored_heatmap, mode="RGBA").save(out_dir / f"{slide_name}_heatmap.png", dpi=(300, 300))

    top_indices = np.argsort(raw_scores)[-10:][::-1] 
    slide_patch_dir = patch_dir / run_name / slide_name.split('.')[0]
    slide_patch_dir.mkdir(parents=True, exist_ok=True)

    slide_pred_text = "Drinking" if logits.softmax(dim=-1)[0, 1].item() >= 0.5 else "Non-Drinking"
    slide_specific_prompt, slide_specific_sim = get_top_specific_prompt(embedding, specific_texts_dict, slide_pred_text, device)

    top_10_log = {
        "slide_level_prediction": slide_pred_text,
        "dominant_specific_slide_phenotype": slide_specific_prompt,
        "dominant_phenotype_similarity": slide_specific_sim,
        "top_10_patches": []
    }

    for rank, idx in enumerate(top_indices):
        x, y = coords_np[idx]
        score_val = float(raw_scores[idx])
        
        patch_feature = adapted_patch_features[idx]
        patch_specific_prompt, patch_specific_sim = get_top_specific_prompt(patch_feature, specific_texts_dict, slide_pred_text, device)
        
        if agg_type == "ABMIL" and draw is not None:
            tx, ty = int(x // scale_factor), int(y // scale_factor)
            outline_width = max(2, patch_size_thumb // 8)
            draw.ellipse([tx, ty, tx + patch_size_thumb, ty + patch_size_thumb], outline="red", width=outline_width)

        img_name = f"Rank{rank+1}_{score_type}_{score_val:.4f}.png"
        patch_img = slide.read_region((int(x), int(y)), 0, (patch_size, patch_size)).convert("RGB")
        patch_img.save(slide_patch_dir / img_name, dpi=(300, 300))
        
        top_10_log["top_10_patches"].append({
            "rank": rank + 1,
            "score_type": score_type,
            "score_value": score_val,
            "specific_patch_phenotype": patch_specific_prompt,
            "phenotype_similarity": patch_specific_sim,
            "coordinates": {"x": int(x), "y": int(y)},
            "image_file": img_name
        })
        
    with open(slide_patch_dir / f"{slide_name.split('.')[0]}_top10.json", 'w') as f:
        json.dump(top_10_log, f, indent=4)
        
    if agg_type == "ABMIL" and circles_img is not None:
        circles_img.save(out_dir / f"{slide_name}_top10_circles.png", dpi=(300, 300))

    if agg_type == "ABMIL" and slide_pred_text == "Drinking":
        _map_adjacent_regions(slide, raw_scores, coords_np, patch_size, specific_texts_dict, slide_pred_text,
                              adapted_patch_features, device, original_thumb, scale_factor, thumb_h, slide_patch_dir, out_dir, slide_name)


def _map_adjacent_regions(slide, raw_scores, coords_np, patch_size, specific_texts_dict, slide_pred_text, 
                          adapted_patch_features, device, original_thumb, scale_factor, thumb_h, slide_patch_dir, out_dir, slide_name):
    """Helper function to cluster high-attention regions and stitch images."""
    predictive_dir = slide_patch_dir / "predictive_drinking_regions"
    predictive_dir.mkdir(parents=True, exist_ok=True)

    num_patches = len(raw_scores)
    top_k = max(1, int(num_patches * 0.05))
    high_score_indices = np.argsort(raw_scores)[-top_k:]

    grid_map = {}
    for idx in high_score_indices:
        x, y = coords_np[idx]
        gx, gy = int(x // patch_size), int(y // patch_size)
        grid_map[(gx, gy)] = (idx, x, y)

    visited = set()
    clusters = []

    for start_node in grid_map.keys():
        if start_node not in visited:
            cluster = []
            queue = [start_node]
            visited.add(start_node)

            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                cx, cy = curr

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        neighbor = (cx + dx, cy + dy)
                        if neighbor in grid_map and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

            if len(cluster) >= 4:
                clusters.append(cluster)

    ranked_clusters = []
    for cluster in clusters:
        cum_attention = sum(raw_scores[grid_map[node][0]] for node in cluster)
        ranked_clusters.append((cum_attention, cluster))

    ranked_clusters.sort(key=lambda x: x[0], reverse=True)

    region_map_img = original_thumb.copy()
    draw_map = ImageDraw.Draw(region_map_img)
    
    try:
        target_font_size = max(16, thumb_h // 40)
        font = ImageFont.truetype("arial.ttf", target_font_size)
    except IOError:
        font = ImageFont.load_default()

    regions_summary_log = []

    for c_idx, (cum_attention, cluster) in enumerate(ranked_clusters):
        cluster_dir = predictive_dir / f"Region_{c_idx+1}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        min_gx, min_gy = min(node[0] for node in cluster), min(node[1] for node in cluster)
        max_gx, max_gy = max(node[0] for node in cluster), max(node[1] for node in cluster)

        region_patch_features = [adapted_patch_features[grid_map[node][0]] for node in cluster]
        region_feature = torch.stack(region_patch_features).mean(dim=0)
        region_prompt, region_sim = get_top_specific_prompt(region_feature, specific_texts_dict, slide_pred_text, device)
        
        regions_summary_log.append({
            "region_rank": c_idx + 1,
            "num_patches": len(cluster),
            "cumulative_attention": float(cum_attention),
            "dominant_specific_phenotype": region_prompt,
            "phenotype_similarity": float(region_sim)
        })

        thumb_min_x = int((min_gx * patch_size) // scale_factor)
        thumb_min_y = int((min_gy * patch_size) // scale_factor)
        thumb_max_x = int(((max_gx + 1) * patch_size) // scale_factor)
        thumb_max_y = int(((max_gy + 1) * patch_size) // scale_factor)

        draw_map.rectangle([thumb_min_x, thumb_min_y, thumb_max_x, thumb_max_y], outline="yellow", width=4)
        text_label = f"Region {c_idx+1}"
        text_y_pos = max(0, thumb_min_y - 30) 
        draw_map.text((thumb_min_x, text_y_pos), text_label, fill="yellow", font=font, stroke_width=2, stroke_fill="black")

        width = (max_gx - min_gx + 1) * patch_size
        height = (max_gy - min_gy + 1) * patch_size
        stitched_img = Image.new('RGB', (width, height), color=(255, 255, 255))

        for node in cluster:
            idx, orig_x, orig_y = grid_map[node]
            patch_img = slide.read_region((int(orig_x), int(orig_y)), 0, (patch_size, patch_size)).convert("RGB")
            patch_img.save(cluster_dir / f"patch_x{int(orig_x)}_y{int(orig_y)}.png", dpi=(300, 300))

            paste_x = (node[0] - min_gx) * patch_size
            paste_y = (node[1] - min_gy) * patch_size
            stitched_img.paste(patch_img, (paste_x, paste_y))

        stitched_filename = f"Region_{c_idx+1}_size{len(cluster)}_CumAttn{cum_attention:.4f}.png"
        stitched_img.save(predictive_dir / stitched_filename, dpi=(300, 300))

    region_map_img.save(out_dir / f"{slide_name}_region_map.png", dpi=(300, 300))
    with open(predictive_dir / "regions_summary.json", 'w') as f:
        json.dump(regions_summary_log, f, indent=4)


def evaluate_split_detailed(model: torch.nn.Module, dataset: WSIDataset, device: torch.device, split_name: str, seed: int) -> tuple[float, float, list]:
    """Runs inference over a dataset split and prints formatted metrics."""
    targets, preds, slide_details = [], [], []
    
    print(f"\n--- {split_name.upper()} SET PREDICTIONS (Seed {seed}) ---")
    print(f"{'Subject ID':<15} | {'Slide Name':<25} | {'GT':<15} | {'Pred':<15} | {'Prob':<5} | {'Match'}")
    print("-" * 85)
    
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            features, label, slide_name, _, subject_id = dataset[i]
            features = features.to(device)
            
            logits, _, _ = model(features)
            prob = logits.softmax(dim=-1)[0, 1].item()
            pred = 1 if prob >= 0.5 else 0
            
            targets.append(label)
            preds.append(pred)
            
            str_gt = "Drinking" if label == 1 else "Non-Drinking"
            str_pred = "Drinking" if pred == 1 else "Non-Drinking"
            match = "✅" if pred == label else "❌"
            
            print(f"{subject_id:<15} | {slide_name:<25} | {str_gt:<15} | {str_pred:<15} | {prob:.2f} | {match}")
            
            slide_details.append({
                "subject_id": subject_id, "slide_name": slide_name,
                "ground_truth": str_gt, "prediction": str_pred,
                "probability": prob, "is_correct": bool(pred == label)
            })
            
    acc = sklearn.metrics.accuracy_score(targets, preds)
    bacc = sklearn.metrics.balanced_accuracy_score(targets, preds)
    return acc, bacc, slide_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=str, choices=["ZSMIL", "TaskRes", "CLIPAdapter"], default="CLIPAdapter")
    parser.add_argument('--encoder', type=str, choices=["PLIP", "CONCH"], default="CONCH") 
    parser.add_argument('--aggregator', type=str, choices=["ABMIL", "MeanPool"], default="MeanPool")
    parser.add_argument('--freeze_scale', action='store_true')
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--generate_visuals', action='store_true', help="Generate heatmaps and top 10 patches")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).resolve().parent
    features_dir = project_root / "outputs" / f"features_{args.encoder}"
    wsi_dir = project_root / "dataset" / "raw_wsis"
    
    freeze_str = "_frozen" if args.freeze_scale else ""
    experiment_type = f"{args.adapter}_{args.aggregator}{freeze_str}_{args.encoder}"
    
    checkpoints_dir = project_root / "outputs" / "checkpoints" / experiment_type
    heatmaps_dir = project_root / "outputs" / "heatmaps" / experiment_type
    patch_dir = project_root / "outputs" / "top_10_analysis" / experiment_type
    inference_stats_dir = project_root / "outputs" / "inference_stats"
    
    for d in [heatmaps_dir, patch_dir, inference_stats_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    text_prototypes = np.load(features_dir / "text_prototypes.npy")
    specific_texts_path = features_dir / "specific_text_features.pt"
    
    if not specific_texts_path.exists():
        logging.error(f"Could not find {specific_texts_path}. Please re-run extract_features.py.")
        return
        
    specific_texts_dict = torch.load(specific_texts_path, map_location="cpu")
    val_dataset, test_dataset = WSIDataset(features_dir, split="val"), WSIDataset(features_dir, split="test")
    
    logging.info(f"\n{'='*85}\n[INFERENCE] {experiment_type}\n{'='*85}")

    val_accs, val_baccs, test_accs, test_baccs = [], [], [], []
    all_seeds_val_details, all_seeds_test_details = [], []
    use_learnable = not args.freeze_scale

    for n_seed in range(args.n_seeds):
        run_name = f"{experiment_type}_seed{n_seed}"
        ckpt_path = checkpoints_dir / f"{run_name}_best.pth"
        
        if not ckpt_path.exists(): 
            logging.warning(f"Checkpoint not found for {run_name}. Skipping.")
            continue

        if args.adapter == "ZSMIL": model = ZSMIL(text_embeddings=text_prototypes, use_learnable_scale=use_learnable, agg_type=args.aggregator)
        elif args.adapter == "TaskRes": model = TaskRes(text_embeddings=text_prototypes, use_learnable_scale=use_learnable, agg_type=args.aggregator)
        elif args.adapter == "CLIPAdapter": model = CLIPAdapter(text_embeddings=text_prototypes, use_learnable_scale=use_learnable, agg_type=args.aggregator)
            
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()

        v_acc, v_bacc, v_details = evaluate_split_detailed(model, val_dataset, device, "Validation", n_seed)
        t_acc, t_bacc, t_details = evaluate_split_detailed(model, test_dataset, device, "Test", n_seed)
        
        val_accs.append(v_acc)
        val_baccs.append(v_bacc)
        test_accs.append(t_acc)
        test_baccs.append(t_bacc)
        
        all_seeds_val_details.append({f"seed_{n_seed}": v_details})
        all_seeds_test_details.append({f"seed_{n_seed}": t_details})
        
        if args.generate_visuals and n_seed == 0:
            logging.info(f"Generating visuals for {run_name} (Test Set)...")
            for i in range(len(test_dataset)):
                features, _, slide_name, coords, _ = test_dataset[i]
                features = features.to(device)
                analyze_visuals(slide_name, features, coords, model, args.aggregator, wsi_dir / slide_name, heatmaps_dir, patch_dir, device, run_name, specific_texts_dict)

    if not test_accs:
        logging.error("No successful inference runs.")
        return

    summary_dict = {
        "experiment": experiment_type,
        "n_seeds_evaluated": len(test_accs),
        "metrics": {
            "validation": {
                "accuracy_mean": float(np.mean(val_accs)), "accuracy_std": float(np.std(val_accs)),
                "balanced_accuracy_mean": float(np.mean(val_baccs)), "balanced_accuracy_std": float(np.std(val_baccs))
            },
            "test": {
                "accuracy_mean": float(np.mean(test_accs)), "accuracy_std": float(np.std(test_accs)),
                "balanced_accuracy_mean": float(np.mean(test_baccs)), "balanced_accuracy_std": float(np.std(test_baccs))
            }
        },
        "subject_level_predictions": {
            "validation_set": all_seeds_val_details,
            "test_set": all_seeds_test_details
        }
    }
    
    json_out_path = inference_stats_dir / f"{experiment_type}_inference_summary.json"
    with open(json_out_path, 'w') as f:
        json.dump(summary_dict, f, indent=4)
    logging.info(f"Detailed inference summary saved to {json_out_path}")

if __name__ == "__main__":
    main()

"""
demo.py
Few-Shot Multiple Instance Learning (MIL) Pipeline.
"""

import argparse
import logging
import os
import re
import sys
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import openslide
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.train_eval import WSIPatchDataset, load_or_create_splits, train_and_eval_fold, plot_and_save_curves
from utils.xai import run_xai_analysis
from utils.roc_plot import save_roc_curve

from utils.mil_models import (
    ABMILAdapterModel, 
    ZSMILModel, 
    TaskResModel, 
    MaxPoolModel, 
    DualAdaptModel,
    NLTaskResModel,
    AffineTaskResModel
)

# ==========================================
# CONFIGURATION & DIRECTORIES
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent
STANDALONE_DIR = PROJECT_ROOT / "outputs"
PATCHES_DIR = STANDALONE_DIR / "patches"
FEATURES_DIR = STANDALONE_DIR / "features"
PREVIEWS_DIR = STANDALONE_DIR / "previews"

for d in [PATCHES_DIR, FEATURES_DIR, PREVIEWS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 224
LEVEL = 0
BATCH_SIZE = 256

CACHE_DIR = PROJECT_ROOT / "models" / "cache_for_HF"
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
if CACHE_DIR.exists() and any(CACHE_DIR.iterdir()):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

PREFIXES = [
    "An H&E stained liver slide showing ",
    "Histopathological examination of a liver slide displaying "
]
HEALTHY_MARKERS = [
    "orderly trabecular architecture.", "clear sinusoids.", 
    "intact portal tracts.", "normal central veins.", "unremarkable parenchyma."
]
ALD_MARKERS = [
    "macrovesicular steatosis.", "hepatocyte ballooning.", 
    "Mallory-Denk bodies.", "neutrophilic infiltrate.", "pericellular fibrosis."
]
BAG_PROMPTS = {
    "Healthy Liver": [p + m for p in PREFIXES for m in HEALTHY_MARKERS], # 10 Prompts
    "Alcoholic Liver Disease": [p + m for p in PREFIXES for m in ALD_MARKERS] # 10 Prompts
}

def set_seed(seed=42):
    """Ensures deterministic behavior across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_subject_id(filename):
    m = re.search(r"Coh[a-zA-Z0-9]+_(\d+)_", filename)
    return m.group(1) if m else "UNKNOWN"

# ==========================================
# TISSUE PROCESSING & PATCH EXTRACTION
# ==========================================
def generate_tissue_mask(slide, thumb_factor=32):
    Wl, Hl = slide.level_dimensions[LEVEL]
    thumb = slide.get_thumbnail((Wl // thumb_factor, Hl // thumb_factor))
    thumb_arr = np.array(thumb.convert("RGB"))

    # Crop thumbnail to only the LEFT half
    half_w_thumb = (Wl // 2) // thumb_factor
    thumb_arr = thumb_arr[:, :half_w_thumb]

    hsv = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(S, (7, 7), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 400 
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    contour_mask = np.zeros_like(binary, dtype=np.uint8)
    cv2.drawContours(contour_mask, valid_contours, -1, 1, thickness=cv2.FILLED)

    kernel = np.ones((5, 5), np.uint8) 
    eroded_macro_mask = cv2.erode(contour_mask, kernel, iterations=4)

    S_float = S.astype(np.float32) / 255.0
    V_float = hsv[:, :, 2].astype(np.float32) / 255.0
    is_actual_tissue = (S_float > 0.10) & (V_float > 0.05)

    final_mask = (eroded_macro_mask == 1) & is_actual_tissue
    return final_mask, thumb_arr, valid_contours
    
def generate_slide_preview(slide, slide_name, subject_id, thumb_factor=32):
    original_path = PREVIEWS_DIR / f"{subject_id}_original.png"
    processed_path = PREVIEWS_DIR / f"{subject_id}_bg_removed.png"
    
    if original_path.exists() and processed_path.exists():
        return
        
    mask, thumb_arr, valid_contours = generate_tissue_mask(slide, thumb_factor)
    
    plt.imsave(original_path, thumb_arr)
    
    processed_thumb = thumb_arr.copy()
    processed_thumb[~mask] = [255, 255, 255] 
    plt.imsave(processed_path, processed_thumb)
    
def _extract_patches_worker(slide_path):
    slide_name = Path(slide_path).name
    subject_id = parse_subject_id(slide_name)
    out_pt = PATCHES_DIR / f"{subject_id}_patches.pt"
    
    slide = openslide.OpenSlide(str(slide_path))
    thumb_factor = 32
    
    if out_pt.exists(): 
        generate_slide_preview(slide, slide_name, subject_id, thumb_factor)
        slide.close()
        return
    
    generate_slide_preview(slide, slide_name, subject_id, thumb_factor)
    mask, _, _ = generate_tissue_mask(slide, thumb_factor)
    
    Wl, Hl = slide.level_dimensions[LEVEL]
    Wl = Wl // 2  
    
    kept_coords = []
    downsample = float(slide.level_downsamples[LEVEL])
    p_size_thumb = PATCH_SIZE // thumb_factor
    
    for j in range(Hl // PATCH_SIZE):
        y0 = int(j * PATCH_SIZE * downsample)
        ty = int(y0 // thumb_factor) 
        
        for i in range(Wl // PATCH_SIZE):   
            x0 = int(i * PATCH_SIZE * downsample)   
            tx = int(x0 // thumb_factor)            
            patch_mask = mask[ty:ty+p_size_thumb, tx:tx+p_size_thumb]
            
            if patch_mask.size == 0 or float(patch_mask.mean()) < 0.75:
                continue 
                
            patch_img = np.array(slide.read_region((x0, y0), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB"))
            gray_patch = cv2.cvtColor(patch_img, cv2.COLOR_RGB2GRAY)
            
            white_pixels = np.sum(gray_patch > 220) 
            total_pixels = PATCH_SIZE * PATCH_SIZE
            if (white_pixels / total_pixels) > 0.25: 
                continue
                
            half = PATCH_SIZE // 2
            focus_scores = [
                cv2.Laplacian(gray_patch[:half, :half], cv2.CV_64F).var(),
                cv2.Laplacian(gray_patch[:half, half:], cv2.CV_64F).var(),
                cv2.Laplacian(gray_patch[half:, :half], cv2.CV_64F).var(),
                cv2.Laplacian(gray_patch[half:, half:], cv2.CV_64F).var() 
            ]
            
            if min(focus_scores) < 250.0: 
                continue
                
            kept_coords.append(torch.tensor([x0, y0]))
    
    if kept_coords:
        torch.save({
            "coords": torch.stack(kept_coords),
            "subject_id": subject_id,
            "slide_name": slide_name
        }, out_pt)
    
    slide.close()

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def run_feature_extraction(device, valid_subjects, subject_meta):
    logging.info("Loading CONCH model for feature extraction...")
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    conch_checkpoint = PROJECT_ROOT / "models" / "CONCH" / "pytorch_model.bin"
    model, processor = create_model_from_pretrained('conch_ViT-B-16', str(conch_checkpoint))
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer()
                
    # Bag-Level Extraction
    bag_feats_list = []
    text_to_feature_dict = {} 
    
    with torch.no_grad():
        for cls in ["Healthy Liver", "Alcoholic Liver Disease"]:
            cls_feats = []
            for prompt in BAG_PROMPTS[cls]:
                inputs = tokenizer([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                outputs = model.encode_text(inputs["input_ids"].to(device))
                feat = F.normalize(outputs, p=2, dim=-1).squeeze(0).cpu().numpy()
                cls_feats.append(feat)
                text_to_feature_dict[prompt] = feat 
                
            if args.text_pool == "mean":
                cls_prototype = np.mean(cls_feats, axis=0)
                bag_feats_list.append(cls_prototype / np.linalg.norm(cls_prototype))
            else:
                for feat in cls_feats: # LSE mode saves all 45 un-averaged features
                    bag_feats_list.append(feat / np.linalg.norm(feat))
    
    np.save(FEATURES_DIR / f"bag_embeddings_{args.text_pool}.npy", np.stack(bag_feats_list))
    torch.save(text_to_feature_dict, FEATURES_DIR / "training_text_embeddings.pt")
    
    extracted_all = all((FEATURES_DIR / f"{s}_features.pt").exists() for s in valid_subjects)
    if extracted_all:
        logging.info("All visual features already extracted. Proceeding to training.")
        return

    raw_wsis_dir = PROJECT_ROOT / "dataset" / "raw_wsis"
    pt_files = [PATCHES_DIR / f"{s}_patches.pt" for s in valid_subjects if (PATCHES_DIR / f"{s}_patches.pt").exists()]
    
    for pt_path in tqdm(pt_files, desc="Extracting Image Features"):
        feature_out_path = FEATURES_DIR / pt_path.name.replace("_patches.pt", "_features.pt")
        if feature_out_path.exists(): continue
            
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        coords = data["coords"]
        subj_id = data["subject_id"]
        slide_filename = data["slide_name"]
        
        slide_path = raw_wsis_dir / slide_filename
        dataset = WSIPatchDataset(slide_path=slide_path, coords=coords, patch_size=PATCH_SIZE, level=LEVEL)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        slide_feats = []
        with torch.no_grad():
            for pixel_values in dataloader:
                pixel_values = pixel_values.to(device)
                with torch.autocast(device_type=device.type):
                    feats = model.encode_image(pixel_values)
                slide_feats.append(F.normalize(feats, p=2, dim=-1).cpu().float().numpy())
                
        torch.save({
            "features": torch.from_numpy(np.concatenate(slide_feats, axis=0)),
            "coords": coords, 
            "subject_id": subj_id,
            "binary_category": "Healthy Liver" if subject_meta[subj_id]["category"] == "ND" else "Alcoholic Liver Disease"
        }, feature_out_path)
        
# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["abmil_adapter", "zsmil", "taskres",
    "affine_taskres", "max_pool", "dual_adapt", "nl_taskres"], default="abmil_adapter")
    parser.add_argument('--alpha', type=float, default=0.25, help="TaskRes alpha parameter to control text adaptation strength.")
    parser.add_argument('--max_pool_head', type=str, choices=["none", "adapter", "taskres"], default="none")
    parser.add_argument('--text_pool', type=str, choices=["mean", "lse"], default="mean", help="aggregate text prompts (mean or log-sum-exp).")
    parser.add_argument('--taskres_drop', type=float, default=0.1, help="Dropout probability for TaskRes parameter.")
    parser.add_argument('--epochs', type=int, default=200, help="Max training epochs.")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate.")
    parser.add_argument('--reduction', type=int, default=8, help="Adapter bottleneck reduction factor.")
    parser.add_argument('--abmil_d', type=int, default=64, help="ABMIL projection dimension.")
    parser.add_argument('--shots', type=int, default=1, help="Samples per class for training.")
    parser.add_argument('--iterations', type=int, default=5, help="Number of random splits.")
    parser.add_argument('--dataset_pct', type=float, default=1.0, help="Percentage of total dataset to use.")
    parser.add_argument('--seeds', type=int, default=1, help="Random seeds per fold.") 
    parser.add_argument('--patience', type=int, default=25, help="Early stopping patience.") 
    args = parser.parse_args()
    
    args.num_healthy_prompts = len(BAG_PROMPTS["Healthy Liver"])
    
    args.accum_steps = args.shots * 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    RAW_SUBJECTS = """
    10095 Rhesus 7a M ND
    10093 Rhesus 7a M ND
    10094 Rhesus 7a M ND
    10096 Rhesus 7a M ND
    10088 Rhesus 7a M HD
    10097 Rhesus 7a M HD
    10091 Rhesus 7a M VHD
    10098 Rhesus 7a M VHD
    10087 Rhesus 7a M BD
    10089 Rhesus 7a M LD
    10090 Rhesus 7a M LD
    10092 Rhesus 7a M LD
    10185 Rhesus 7b M LD
    10182 Rhesus 7b M ND
    10085 Rhesus 7b M LD
    10083 Rhesus 7b M LD
    10183 Rhesus 7b M ND
    10082 Rhesus 7b M HD
    10184 Rhesus 7b M ND
    10084 Rhesus 7b M LD
    10086 Rhesus 7b M BD
    10079 Rhesus 6a F VHD
    10080 Rhesus 6a F VHD
    10069 Rhesus 6a F VHD
    10068 Rhesus 6a F ND
    10070 Rhesus 6a F VHD
    10076 Rhesus 6a F ND
    10071 Rhesus 6a F ND
    10078 Rhesus 6a F VHD
    10188 Rhesus 6b F ND
    10186 Rhesus 6b F ND
    10187 Rhesus 6b F ND
    10077 Rhesus 6b F LD
    10075 Rhesus 6b F HD
    10074 Rhesus 6b F LD
    10073 Rhesus 6b F VHD
    10072 Rhesus 6b F LD
    """

    SUBJECT_META, ND_POOL, D_POOL = {}, [], []
    for line in RAW_SUBJECTS.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 5:
            subj_id, cohort, cat = parts[0], parts[2], parts[4]
            SUBJECT_META[subj_id] = {"category": cat, "cohort": cohort}
            if cat == "ND": ND_POOL.append(subj_id)
            else: D_POOL.append(subj_id)

    if args.dataset_pct < 1.0:
        rng = random.Random(42)  
        nd_keep = max(1, int(len(ND_POOL) * args.dataset_pct))
        d_keep = max(1, int(len(D_POOL) * args.dataset_pct))
        ND_POOL = rng.sample(ND_POOL, nd_keep)
        D_POOL = rng.sample(D_POOL, d_keep)
        logging.info(f"Subsampled Dataset ({args.dataset_pct*100}%): {nd_keep} ND and {d_keep} D subjects.")

    valid_subjects = set(ND_POOL + D_POOL)
    selected_svs_filenames = [f"Coh{SUBJECT_META[s]['cohort']}_{s}_Nx_Liver_HE.svs" for s in valid_subjects]
    raw_wsis = {parse_subject_id(f): PROJECT_ROOT / "dataset" / "raw_wsis" / f for f in selected_svs_filenames}

    SHARED_SPLITS_DIR = STANDALONE_DIR / "shared_splits"
    SHARED_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON_PATH = SHARED_SPLITS_DIR / f"dataset_splits_{args.shots}shot_{args.iterations}iter_{int(args.dataset_pct*100)}pct.json"
    
    if args.model == "abmil_adapter": model_str = f"{args.model}_r{args.reduction}_d{args.abmil_d}_{args.text_pool}"
    elif args.model == "dual_adapt": model_str = f"{args.model}_r{args.reduction}_a{args.alpha}_d{args.abmil_d}_{args.text_pool}"
    elif args.model == "taskres": model_str = f"{args.model}_a{args.alpha}_d{args.abmil_d}_{args.text_pool}"
    elif args.model == "nl_taskres": model_str = f"{args.model}_r{args.reduction}_a{args.alpha}_d{args.abmil_d}_{args.text_pool}"
    elif args.model == "affine_taskres": model_str = f"{args.model}_a{args.alpha}_d{args.abmil_d}_{args.text_pool}"
    elif args.model == "max_pool": model_str = f"{args.model}_{args.max_pool_head}_r{args.reduction}_{args.text_pool}"
    else: model_str = f"{args.model}_d{args.abmil_d}_{args.text_pool}"

    DYNAMIC_RESULTS_DIR = STANDALONE_DIR / f"{model_str}_{args.shots}shot_{args.iterations}iter"

    if (DYNAMIC_RESULTS_DIR / "summary_metrics.txt").exists():
        logging.info(f"Experiment {DYNAMIC_RESULTS_DIR.name} already exists and is complete. Skipping.")
        sys.exit(0)
    
    PROBS_DIR = DYNAMIC_RESULTS_DIR / "test_probabilities" 
    CURVES_DIR = DYNAMIC_RESULTS_DIR / "training_curves" 
    
    DYNAMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True) 

    logging.info("Starting Processing Pipeline...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(tqdm(ex.map(_extract_patches_worker, list(raw_wsis.values())), total=len(raw_wsis)))
        
    run_feature_extraction(device, valid_subjects, SUBJECT_META)
    bag_embeddings = np.load(FEATURES_DIR / f"bag_embeddings_{args.text_pool}.npy")
    instance_embeddings = None  
    
    flat_val_baccs, flat_test_baccs, flat_test_aucs, flat_test_f1s, flat_epochs = [], [], [], [], []
    combo_report_logs, xai_infos = [], []
    
    dataset_splits = load_or_create_splits(SPLITS_JSON_PATH, args.shots, ND_POOL, D_POOL, SUBJECT_META, args.iterations)
    
    logging.info(f"Initiating Few-Shot Evaluation: {args.model.upper()} | {args.shots} Shots")
    
    for fold_idx, split_info in enumerate(dataset_splits, start=1):
        fold_val_bacc, fold_test_bacc, fold_epochs = [], [], []
        train_ids = split_info["train_nd"] + split_info["train_d"]
        val_ids = split_info["val_nd"] + split_info["val_d"]
        test_ids = split_info["test_ids"]
        
        for n_seed in range(args.seeds):
            set_seed(n_seed)

            val_bacc, test_bacc, test_auc, test_f1, stopped_epoch, eval_probs, eval_preds, eval_ids, b_weights, history = train_and_eval_fold(
                train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
                bag_embeddings=bag_embeddings, instance_embeddings=instance_embeddings, device=device, args=args,
                features_dir=FEATURES_DIR
            )

            # --- Probability Reporting ---
            txt_lines = [
                "============================================================",
                "               EVALUATION PROBABILITY REPORT",
                "============================================================",
                f"Iteration : {fold_idx}",
                f"Seed      : {n_seed}",
                f"Model     : {args.model.upper()}",
                f"Shots     : {args.shots}",
                "------------------------------------------------------------",
                f"Train Subjects: {', '.join(train_ids)}",
                f"Val Subjects  : {', '.join(val_ids)}",
                "============================================================",
                f"{'Subject ID':<15} | {'True Label':<15} | {'Pred Label':<15} | {'Correct':<8} | {'Prob(Drinking)':<15} | {'Confidence'}",
                "------------------------------------------------------------"
            ]
            
            for subj in eval_ids:
                true_label = 1 if subj in D_POOL else 0
                pred_label = eval_preds.get(subj, -1)
                prob = eval_probs.get(subj, -1.0)
                if prob == -1.0: conf_str = "ERROR"
                elif prob >= 0.8 or prob <= 0.2: conf_str = "High"
                elif prob >= 0.6 or prob <= 0.4: conf_str = "Medium"
                else: conf_str = "Low"

                t_class = "ALD" if true_label == 1 else "Healthy"
                p_class = "ALD" if pred_label == 1 else "Healthy"
                correct = "Yes" if true_label == pred_label else "No"
                
                txt_lines.append(f"{subj:<15} | {t_class:<15} | {p_class:<15} | {correct:<8} | {prob:<15.4f} | {conf_str}")

            txt_lines.append("============================================================")
            
            with open(PROBS_DIR / f"iter{fold_idx:02d}_seed{n_seed}.txt", "w") as f:
                f.write("\n".join(txt_lines))

            plot_and_save_curves(history, CURVES_DIR, fold_idx, n_seed)

            test_targets = [1 if subj in D_POOL else 0 for subj in test_ids]
            test_probs_list = [eval_probs.get(subj, 0.0) for subj in test_ids]
            
            roc_save_path = CURVES_DIR / f"roc_curve_iter{fold_idx:02d}_seed{n_seed}.png"
            save_roc_curve(
                targets=test_targets, 
                probs=test_probs_list, 
                save_path=roc_save_path, 
                title_prefix=f"Iter {fold_idx} (Seed {n_seed})"
            )

            fold_val_bacc.append(val_bacc)
            fold_test_bacc.append(test_bacc)
            fold_epochs.append(stopped_epoch)
            
            flat_val_baccs.append(val_bacc)
            flat_test_baccs.append(test_bacc)
            flat_test_aucs.append(test_auc)
            flat_test_f1s.append(test_f1)
            flat_epochs.append(stopped_epoch)
            
            if fold_idx in [1, 2] and n_seed == 0: 
                xai_count = 0
                for subj_id in eval_ids:
                    if subj_id in D_POOL and eval_preds.get(subj_id) == 1:
                        xai_infos.append({
                            "wsi_id": subj_id, 
                            "combo_idx": fold_idx, 
                            "seed_idx": n_seed, 
                            "weights": b_weights
                        })
                        xai_count += 1  
                        
                        if xai_count >= 5:
                            break
        
        avg_val_bacc, std_val_bacc = np.mean(fold_val_bacc), np.std(fold_val_bacc)
        avg_test_bacc, std_test_bacc = np.mean(fold_test_bacc), np.std(fold_test_bacc)
        avg_epochs, std_epochs = np.mean(fold_epochs), np.std(fold_epochs)
        
        log_str = (f"Iter {fold_idx:02d}/{args.iterations} | Train:[{','.join(train_ids)}] "
                   f"Val:[{','.join(val_ids)}] Test:[{len(test_ids)} subjects] | "
                   f"Epochs: {avg_epochs:.1f}±{std_epochs:.1f} | "
                   f"Val BACC: {avg_val_bacc:.4f}±{std_val_bacc:.4f} | "
                   f"Test BACC: {avg_test_bacc:.4f}±{std_test_bacc:.4f}")
        logging.info(log_str)
        combo_report_logs.append(f"Iter {fold_idx:02d}: Train:[{','.join(train_ids)}] | Test Subjects: [{len(test_ids)}] -> Test BACC: {avg_test_bacc:.4f} ± {std_test_bacc:.4f}")
            
            
    if args.model == "abmil_adapter": tmp_model = ABMILAdapterModel(text_embeddings=bag_embeddings, reduction=args.reduction, abmil_d=args.abmil_d, text_pool=args.text_pool)
    elif args.model == "zsmil": tmp_model = ZSMILModel(text_embeddings=bag_embeddings, abmil_d=args.abmil_d, text_pool=args.text_pool)
    elif args.model == "taskres": tmp_model = TaskResModel(text_embeddings=bag_embeddings, alpha=args.alpha, abmil_d=args.abmil_d, text_pool=args.text_pool)
    elif args.model == "nl_taskres": tmp_model = NLTaskResModel(text_embeddings=bag_embeddings, alpha=args.alpha, reduction=args.reduction, abmil_d=args.abmil_d, text_pool=args.text_pool)
    elif args.model == "affine_taskres": tmp_model = AffineTaskResModel(text_embeddings=bag_embeddings, alpha=args.alpha, abmil_d=args.abmil_d, text_pool=args.text_pool)
    elif args.model == "dual_adapt": tmp_model = DualAdaptModel(text_embeddings=bag_embeddings, reduction=args.reduction, alpha=args.alpha, abmil_d=args.abmil_d, text_pool=args.text_pool)
    elif args.model == "max_pool": tmp_model = MaxPoolModel(text_embeddings=bag_embeddings, head_type=args.max_pool_head, reduction=args.reduction, alpha=args.alpha)
    else: tmp_model = None

    trainable_params = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad) if tmp_model else 0
            
    summary = (
        f"\n{'='*80}\nFEW-SHOT EVALUATION SUMMARY (Model: {args.model.upper()} - {args.shots} Shot)\n{'='*80}\n"
        f"Average Epochs: {np.mean(flat_epochs):.1f} ± {np.std(flat_epochs):.1f}\n{'='*80}\n"
        f"INDIVIDUAL COMBO DETAILS:\n" + "\n".join(combo_report_logs) + f"\n{'='*80}\n"
        f"OVERALL METRICS (Across {args.iterations * args.seeds} Total Runs):\n"
        f"Trainable Parameters: {trainable_params}\n"  # <-- THIS WAS MISSING
        f"Validation BACC: {np.mean(flat_val_baccs):.4f} ± {np.std(flat_val_baccs):.4f}\n"
        f"Test BACC:       {np.mean(flat_test_baccs):.4f} ± {np.std(flat_test_baccs):.4f}\n"
        f"Test AUC-ROC:    {np.mean(flat_test_aucs):.4f} ± {np.std(flat_test_aucs):.4f}\n"
        f"Test F1-Score:   {np.mean(flat_test_f1s):.4f} ± {np.std(flat_test_f1s):.4f}\n{'='*80}"
    )
    print(summary)
    with open(DYNAMIC_RESULTS_DIR / f"summary_metrics.txt", "w") as f: f.write(summary)
    
    if xai_infos:
        run_xai_analysis(
            xai_infos=xai_infos, 
            args=args, 
            bag_embeddings=bag_embeddings,
            instance_embeddings=instance_embeddings,
            device=device, 
            features_dir=FEATURES_DIR, 
            raw_wsis_map=raw_wsis, 
            results_dir=DYNAMIC_RESULTS_DIR, 
            patch_size=PATCH_SIZE, 
            level=LEVEL
        )

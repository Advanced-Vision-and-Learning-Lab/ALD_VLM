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
import json
import importlib.metadata
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
from utils.conch_features import FEATURE_VERSION, tokenize_conch, sha256_file
from utils.mil_models import BASELINE_MODELS, build_model
from utils.augmentation import AUGMENTATION_VERSION, extract_augmented_features

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
FEATURES_DIR = STANDALONE_DIR / "features_conch_v2"
PREVIEWS_DIR = STANDALONE_DIR / "previews"

for d in [PATCHES_DIR, FEATURES_DIR, PREVIEWS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 224
LEVEL = 0
BATCH_SIZE = 32

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
    else:
        slide.close()
        raise ValueError(f"No patches passed tissue/focus QC for subject {subject_id}.")
    
    slide.close()

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def run_feature_extraction(device, valid_subjects, subject_meta, text_pool):
    conch_checkpoint = PROJECT_ROOT / "models" / "CONCH" / "pytorch_model.bin"
    manifest_path = FEATURES_DIR / "extraction_manifest.json"
    expected = {
        "feature_version": FEATURE_VERSION,
        "checkpoint_sha256": sha256_file(conch_checkpoint),
        "prompts": BAG_PROMPTS,
        "patch_size": PATCH_SIZE,
        "level": LEVEL,
        "image_projection": "contrastive",
        "token_length": 128,
        "patch_coordinate_sha256": {s: sha256_file(PATCHES_DIR / f"{s}_patches.pt") for s in valid_subjects},
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError("Feature cache contract changed; move features_conch_v2 aside before extracting.")
        if (all((FEATURES_DIR / f"{s}_features.pt").exists() for s in valid_subjects)
                and (FEATURES_DIR / f"bag_embeddings_{text_pool}.npy").exists()
                and (FEATURES_DIR / "training_text_embeddings.pt").exists()):
            logging.info("Reusing validated CONCH v2 feature cache.")
            return
    elif any(FEATURES_DIR.glob("*_features.pt")):
        raise ValueError("Feature files have no extraction manifest; refusing an unverified cache.")
    logging.info("Loading CONCH model for feature extraction...")
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    model, processor = create_model_from_pretrained('conch_ViT-B-16', str(conch_checkpoint))
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer()
    expected["preprocess"] = str(processor)
    expected["torch_version"] = torch.__version__
    expected["transformers_version"] = importlib.metadata.version("transformers")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(expected, indent=2) + "\n")
                
    # Bag-Level Extraction
    bag_feats_list = []
    text_to_feature_dict = {} 
    
    with torch.no_grad():
        for cls in ["Healthy Liver", "Alcoholic Liver Disease"]:
            cls_feats = []
            for prompt in BAG_PROMPTS[cls]:
                inputs = tokenize_conch(tokenizer, [prompt])
                outputs = model.encode_text(inputs.to(device))
                feat = F.normalize(outputs, p=2, dim=-1).squeeze(0).cpu().numpy()
                cls_feats.append(feat)
                text_to_feature_dict[prompt] = feat 
                
            if text_pool == "mean":
                cls_prototype = np.mean(cls_feats, axis=0)
                bag_feats_list.append(cls_prototype / np.linalg.norm(cls_prototype))
            else:
                for feat in cls_feats:
                    bag_feats_list.append(feat / np.linalg.norm(feat))
    
    np.save(FEATURES_DIR / f"bag_embeddings_{text_pool}.npy", np.stack(bag_feats_list))
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
        if not len(coords):
            raise ValueError(f"No retained patches for {subj_id}.")
        dataset = WSIPatchDataset(slide_path=slide_path, coords=coords, patch_size=PATCH_SIZE, level=LEVEL, preprocess=processor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4 if device.type == "cuda" else 0,
                                pin_memory=device.type == "cuda")
        
        slide_feats = []
        with torch.no_grad():
            for pixel_values in dataloader:
                pixel_values = pixel_values.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    feats = model.encode_image(pixel_values, proj_contrast=True, normalize=True)
                slide_feats.append(F.normalize(feats, p=2, dim=-1).cpu().float().numpy())
                
        torch.save({
            "features": torch.from_numpy(np.concatenate(slide_feats, axis=0)),
            "coords": coords, 
            "subject_id": subj_id,
            "binary_category": "Non-Drinking" if subject_meta[subj_id]["category"] == "ND" else "Drinking",
            "feature_version": FEATURE_VERSION,
        }, feature_out_path)
        
# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["abmil_adapter", "zsmil", "taskres",
    "affine_taskres", "max_pool", "dual_adapt", "nl_taskres", *BASELINE_MODELS], default="abmil_adapter")
    parser.add_argument('--alpha', type=float, default=0.25, help="TaskRes alpha parameter to control text adaptation strength.")
    parser.add_argument('--max_pool_head', type=str, choices=["none", "adapter", "taskres"], default="none")
    parser.add_argument('--text_pool', type=str, choices=["mean", "lse"], default="mean", help="aggregate text prompts (mean or log-sum-exp).")
    parser.add_argument('--taskres_drop', type=float, default=0.1, help="Dropout probability for TaskRes parameter.")
    parser.add_argument('--epochs', type=int, default=200, help="Max training epochs.")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate.")
    parser.add_argument('--reduction', type=int, default=8, help="Adapter bottleneck reduction factor.")
    parser.add_argument('--abmil_d', type=int, default=64, help="ABMIL projection dimension.")
    parser.add_argument('--shots', type=int, default=2, help="Samples per class for training.")
    parser.add_argument('--iterations', type=int, default=5, help="Number of random splits.")
    parser.add_argument('--dataset_pct', type=float, default=1.0, help="Percentage of total dataset to use.")
    parser.add_argument('--seeds', type=int, default=1, help="Random seeds per fold.") 
    parser.add_argument('--patience', type=int, default=25, help="Early stopping patience.") 
    parser.add_argument('--results_root', type=Path, default=STANDALONE_DIR / "revalidated")
    parser.add_argument('--skip_xai', action='store_true', help="Skip optional WSI visualizations.")
    parser.add_argument('--augment', action='store_true', help="Use rotated/flipped and mildly stain-varied images for TRAINING subjects only.")
    parser.add_argument('--aug_views', type=int, default=2, help="Number of cached augmented versions per training slide, in addition to the original.")
    parser.add_argument('--aug_seed', type=int, default=42, help="Reproducible image-augmentation seed.")
    parser.add_argument('--baseline_c', type=float, default=1.0, help="Inverse L2 regularization strength for the two logistic baselines.")
    args = parser.parse_args()
    if (min(args.epochs, args.shots, args.iterations, args.seeds, args.patience, args.reduction, args.abmil_d) < 1
            or args.aug_views < 1 or args.baseline_c <= 0
            or not 0 < args.dataset_pct <= 1 or args.lr <= 0 or not 0 <= args.taskres_drop < 1):
        parser.error("Counts/lr must be positive, dataset_pct in (0,1], and dropout in [0,1).")
    
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
            SUBJECT_META[subj_id] = {"category": cat, "cohort": cohort, "sex": parts[3]}
            if cat == "ND": ND_POOL.append(subj_id)
            else: D_POOL.append(subj_id)

    if args.dataset_pct < 1.0:
        rng = random.Random(42)  
        nd_keep = max(1, int(len(ND_POOL) * args.dataset_pct))
        d_keep = max(1, int(len(D_POOL) * args.dataset_pct))
        ND_POOL = rng.sample(ND_POOL, nd_keep)
        D_POOL = rng.sample(D_POOL, d_keep)
        logging.info(f"Subsampled Dataset ({args.dataset_pct*100}%): {nd_keep} ND and {d_keep} D subjects.")

    valid_subjects = sorted(ND_POOL + D_POOL)
    selected_svs_filenames = [f"Coh{SUBJECT_META[s]['cohort']}_{s}_Nx_Liver_HE.svs" for s in valid_subjects]
    raw_wsis = {parse_subject_id(f): PROJECT_ROOT / "dataset" / "raw_wsis" / f for f in selected_svs_filenames}

    SHARED_SPLITS_DIR = STANDALONE_DIR / "shared_splits"
    SHARED_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON_PATH = SHARED_SPLITS_DIR / f"dataset_splits_{args.shots}shot_{args.iterations}iter_{int(args.dataset_pct*100)}pct.json"
    
    if args.model == "mean_logreg": model_str = f"mean_logreg_c{args.baseline_c:g}"
    elif args.model == "score_calibration": model_str = f"score_calibration_c{args.baseline_c:g}_{args.text_pool}"
    elif args.model == "abmil_adapter": model_str = f"{args.model}_r{args.reduction}_d{args.abmil_d}_{args.text_pool}"
    elif args.model == "dual_adapt": model_str = f"{args.model}_r{args.reduction}_a{args.alpha}_d{args.abmil_d}_{args.text_pool}"
    elif args.model == "taskres": model_str = f"{args.model}_a{args.alpha}_d{args.abmil_d}_{args.text_pool}"
    elif args.model == "nl_taskres": model_str = f"{args.model}_r{args.reduction}_a{args.alpha}_d{args.abmil_d}_{args.text_pool}"
    elif args.model == "affine_taskres": model_str = f"{args.model}_a{args.alpha}_d{args.abmil_d}_{args.text_pool}"
    elif args.model == "max_pool": model_str = f"{args.model}_{args.max_pool_head}_r{args.reduction}_{args.text_pool}"
    else: model_str = f"{args.model}_d{args.abmil_d}_{args.text_pool}"

    dataset_splits = load_or_create_splits(SPLITS_JSON_PATH, args.shots, ND_POOL, D_POOL, SUBJECT_META, args.iterations)
    data_condition = "augmented" if args.augment else "raw"
    DYNAMIC_RESULTS_DIR = args.results_root.resolve() / data_condition / f"{model_str}_{args.shots}shot_{args.iterations}iter"
    augmentation_dir = (FEATURES_DIR.parent / f"{FEATURES_DIR.name}_aug{args.aug_views}_seed{args.aug_seed}") if args.augment else None
    run_config = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "feature_version": FEATURE_VERSION,
        "data_condition": data_condition,
        "augmentation_version": AUGMENTATION_VERSION if args.augment else None,
        "split_sha256": sha256_file(SPLITS_JSON_PATH),
        "checkpoint_sha256": sha256_file(PROJECT_ROOT / "models/CONCH/pytorch_model.bin"),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "sources": {name: sha256_file(PROJECT_ROOT / name) for name in
                    ("demo.py", "utils/train_eval.py", "utils/mil_models.py", "utils/conch_features.py", "utils/xai.py", "utils/augmentation.py", "utils/baselines.py")},
        "subjects": SUBJECT_META,
    }
    config_path = DYNAMIC_RESULTS_DIR / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError("Existing run has a different configuration/code; use a new --results_root.")

    if (DYNAMIC_RESULTS_DIR / "summary_metrics.txt").exists() and config_path.exists():
        logging.info(f"Experiment {DYNAMIC_RESULTS_DIR.name} already exists and is complete. Skipping.")
        sys.exit(0)
    
    PROBS_DIR = DYNAMIC_RESULTS_DIR / "test_probabilities" 
    CURVES_DIR = DYNAMIC_RESULTS_DIR / "training_curves" 
    
    DYNAMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True) 
    config_path.write_text(json.dumps(run_config, indent=2) + "\n")

    logging.info("Starting Processing Pipeline...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(tqdm(ex.map(_extract_patches_worker, list(raw_wsis.values())), total=len(raw_wsis)))
        
    run_feature_extraction(device, valid_subjects, SUBJECT_META, args.text_pool)
    if args.augment:
        # A subject may train in one split and test in another; only the fold's
        # training loader can read these cached views. Evaluation always reads raw.
        augmentation_subjects = sorted({sid for split in dataset_splits for sid in split["train_nd"] + split["train_d"]})
        extract_augmented_features(
            device, augmentation_subjects, FEATURES_DIR, PATCHES_DIR,
            PROJECT_ROOT / "dataset/raw_wsis", PROJECT_ROOT / "models/CONCH/pytorch_model.bin",
            augmentation_dir, num_views=args.aug_views, seed=args.aug_seed, batch_size=BATCH_SIZE,
        )
    bag_embeddings = np.load(FEATURES_DIR / f"bag_embeddings_{args.text_pool}.npy")
    instance_embeddings = None  
    
    flat_val_baccs, flat_test_baccs, flat_test_aucs, flat_test_f1s, flat_epochs = [], [], [], [], []
    combo_report_logs, xai_infos = [], []
    run_metrics = []
    
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
                features_dir=FEATURES_DIR, augmentation_dir=augmentation_dir,
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
                f"{'Subject ID':<15} | {'Partition':<10} | {'True Label':<15} | {'Pred Label':<15} | {'Correct':<8} | {'Prob(Drinking)':<15} | {'Confidence'}",
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

                partition = "validation" if subj in val_ids else "test"
                t_class = "Drinking" if true_label == 1 else "Non-Drinking"
                p_class = "Drinking" if pred_label == 1 else "Non-Drinking"
                correct = "Yes" if true_label == pred_label else "No"
                
                txt_lines.append(f"{subj:<15} | {partition:<10} | {t_class:<15} | {p_class:<15} | {correct:<8} | {prob:<15.4f} | {conf_str}")

            txt_lines.append("============================================================")
            test_confusion = {
                "true_positives": sum(sid in D_POOL and eval_preds[sid] == 1 for sid in test_ids),
                "false_negatives": sum(sid in D_POOL and eval_preds[sid] == 0 for sid in test_ids),
                "false_positives": sum(sid in ND_POOL and eval_preds[sid] == 1 for sid in test_ids),
                "true_negatives": sum(sid in ND_POOL and eval_preds[sid] == 0 for sid in test_ids),
            }
            txt_lines.extend([
                f"TEST drinking subjects correctly identified: {test_confusion['true_positives']}",
                f"TEST drinking subjects missed: {test_confusion['false_negatives']}",
                f"TEST controls incorrectly called drinking: {test_confusion['false_positives']}",
                "Decision threshold: Prob(Drinking) >= 0.5",
            ])
            calibration_details = None
            if args.model == "score_calibration":
                weight = float(b_weights["classifier.weight"].item())
                calibration_details = {
                    "weight": weight, "bias": float(b_weights["classifier.bias"].item()),
                    "score_center": float(b_weights["feature_center"].item()),
                    "score_scale": float(b_weights["feature_scale"].item()),
                    "direction_learned_from_training": "opposite" if weight < 0 else "same" if weight > 0 else "constant",
                }
                txt_lines.append(f"Score direction learned from TRAINING subjects: {calibration_details['direction_learned_from_training']}")
            
            with open(PROBS_DIR / f"iter{fold_idx:02d}_seed{n_seed}.txt", "w") as f:
                f.write("\n".join(txt_lines))

            predictions = [
                {"subject_id": sid, "partition": "validation" if sid in val_ids else "test",
                 "target": int(sid in D_POOL), "prediction": eval_preds[sid],
                 "prob_drinking": eval_probs[sid], **SUBJECT_META[sid]}
                for sid in eval_ids
            ]
            record = {
                "iteration": fold_idx, "seed": n_seed, "train_ids": train_ids,
                "val_ids": val_ids, "test_ids": test_ids, "predictions": predictions,
                "metrics": {"val_bacc": val_bacc, "test_bacc": test_bacc,
                            "test_auc": test_auc, "test_f1": test_f1,
                            "reversed_test_auc_diagnostic": 1.0 - test_auc},
                "best_epoch": int(np.argmin(history["val_loss"]) + 1),
                "stopped_epoch": stopped_epoch, "history": history,
                "data_condition": data_condition,
                "class_counts": {phase: {"non_drinking": sum(sid in ND_POOL for sid in ids),
                                         "drinking": sum(sid in D_POOL for sid in ids)}
                                 for phase, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids))},
                "test_confusion": test_confusion,
                "calibration": calibration_details,
            }
            (PROBS_DIR / f"iter{fold_idx:02d}_seed{n_seed}.json").write_text(json.dumps(record, indent=2) + "\n")
            checkpoint_dir = DYNAMIC_RESULTS_DIR / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(b_weights, checkpoint_dir / f"iter{fold_idx:02d}_seed{n_seed}.pt")
            run_metrics.append({"iteration": fold_idx, "seed": n_seed, **record["metrics"]})

            plot_and_save_curves(history, CURVES_DIR, fold_idx, n_seed)

            test_targets = [1 if subj in D_POOL else 0 for subj in test_ids]
            test_probs_list = [eval_probs[subj] for subj in test_ids]
            
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
            
            # Choose test examples before looking at predictions; include both outcomes/classes.
            if fold_idx == 1 and n_seed == 0 and not args.skip_xai and args.model not in ("max_pool", *BASELINE_MODELS):
                for subj_id in sorted(split_info["test_nd"])[:2] + sorted(split_info["test_d"])[:2]:
                    xai_infos.append({"wsi_id": subj_id, "combo_idx": fold_idx,
                                      "seed_idx": n_seed, "weights": b_weights})

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
            
            
    tmp_model = build_model(args, bag_embeddings)

    trainable_params = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad) if tmp_model else 0
            
    summary = (
        f"\n{'='*80}\nFEW-SHOT EVALUATION SUMMARY (Model: {args.model.upper()} - {args.shots} Shot)\n{'='*80}\n"
        f"Average Epochs: {np.mean(flat_epochs):.1f} ± {np.std(flat_epochs):.1f}\n{'='*80}\n"
        f"INDIVIDUAL COMBO DETAILS:\n" + "\n".join(combo_report_logs) + f"\n{'='*80}\n"
        f"OVERALL METRICS (Across {args.iterations * args.seeds} Total Runs; descriptive SD, overlapping subjects):\n"
        f"Trainable Parameters: {trainable_params}\n"  # <-- THIS WAS MISSING
        f"Validation BACC: {np.mean(flat_val_baccs):.4f} ± {np.std(flat_val_baccs):.4f}\n"
        f"Test BACC:       {np.mean(flat_test_baccs):.4f} ± {np.std(flat_test_baccs):.4f}\n"
        f"Test AUC-ROC:    {np.mean(flat_test_aucs):.4f} ± {np.std(flat_test_aucs):.4f}\n"
        f"Test F1-Score:   {np.mean(flat_test_f1s):.4f} ± {np.std(flat_test_f1s):.4f}\n{'='*80}"
    )
    (DYNAMIC_RESULTS_DIR / "run_metrics.json").write_text(json.dumps(run_metrics, indent=2) + "\n")
    print(summary)
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
    # Write the completion marker only after every requested artifact succeeds.
    summary_path = DYNAMIC_RESULTS_DIR / "summary_metrics.txt"
    temporary_summary = summary_path.with_suffix(".tmp")
    temporary_summary.write_text(summary)
    temporary_summary.replace(summary_path)

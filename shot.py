"""
Few-Shot Evaluation Script.
"""
import argparse
import logging
import os
import re
import json
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sklearn.metrics
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import scipy.ndimage

import cv2

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
S_MAX = 0.10  
V_MIN = 0.05  
BATCH_SIZE = 256
PATIENCE = 50

CACHE_DIR = PROJECT_ROOT / "models" / "cache_for_HF"
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
if CACHE_DIR.exists() and any(CACHE_DIR.iterdir()):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# ARCHITECTURE DEFINITIONS
# ==========================================

class ABMIL(nn.Module):
    def __init__(self, L=512, D=64, K=1):
        super().__init__()
        self.L, self.D, self.K = L, D, K
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights = nn.Linear(D, K)
        self.temperature = nn.Parameter(torch.ones(1))
        self._initialize_weights()

    def _initialize_weights(self):
            nn.init.xavier_uniform_(self.attention_V[0].weight)
            nn.init.zeros_(self.attention_V[0].bias)
            
            nn.init.xavier_uniform_(self.attention_U[0].weight)
            nn.init.zeros_(self.attention_U[0].bias)
            
            nn.init.xavier_uniform_(self.attention_weights.weight)
            if self.attention_weights.bias is not None:
                nn.init.zeros_(self.attention_weights.bias)

    def forward(self, features):
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        
        raw_scores = self.attention_weights(A_V * A_U)
        safe_temp = torch.clamp(self.temperature, min=1e-4)
        w = torch.softmax(raw_scores / safe_temp, dim=0)
        
        #w = torch.softmax(self.attention_weights(A_V * A_U), dim=0)
        return torch.mm(w.t(), features), w


class ABMILAdapterModel(nn.Module):
    def __init__(self, text_embeddings, reduction=4, abmil_d=64):
        super().__init__()
        self.c_in, self.reduction, self.ratio = 512, reduction, 0.2
        self.aggregator = ABMIL(L=self.c_in, D=abmil_d)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        
        self.adapter = nn.Sequential(
            nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
            nn.LayerNorm(self.c_in // self.reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
        )
        nn.init.zeros_(self.adapter[4].weight)

    def forward(self, features):
        embedding, w = self.aggregator(features)
        res = self.adapter(embedding)
        adapted_emb = self.ratio * res + (1 - self.ratio) * embedding
        adapted_emb = F.normalize(adapted_emb, p=2, dim=-1)
        
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (adapted_emb @ prototype_norm.t()) * clamped_scale
        return logits, adapted_emb, w
        
class TGDFMILModel(nn.Module):
    """
    Text-Guided Dynamic Filtering MIL (Inspired by HiVE-MIL NeurIPS 2025).
    Filters out semantically irrelevant patches using text-wise dynamic thresholds,
    then aggregates the SURVIVING 512-d features.
    """
    def __init__(self, pure_concept_embeddings_dict, text_embeddings, reduction=4, abmil_d=64):
        super().__init__()
        self.c_in = 512
        
        # 1. Pure concepts for Text-Guided Filtering
        phrases = sorted(pure_concept_embeddings_dict.keys())
        matrix = np.stack([pure_concept_embeddings_dict[p] for p in phrases])
        self.register_buffer("concept_matrix", torch.tensor(matrix, dtype=torch.float32))
        
        # 2. ABMIL Aggregator for surviving patches
        self.aggregator = ABMIL(L=self.c_in, D=abmil_d)
        
        # 3. Slide-Level Text Prototypes for Classification (Adapter Framework)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        
        self.reduction = reduction
        self.ratio = 0.2
        self.adapter = nn.Sequential(
            nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
            nn.LayerNorm(self.c_in // self.reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
        )
        nn.init.zeros_(self.adapter[4].weight)

    def forward(self, features):
        N = features.shape[0]
        
        # --- STAGE 1: TEXT-GUIDED DYNAMIC FILTERING (TGDF) ---
        v_norm = F.normalize(features, p=2, dim=-1)
        t_norm = F.normalize(self.concept_matrix, p=2, dim=-1)
        
        S = v_norm @ t_norm.t() # [N, Total_Concepts]
        
        # Text-wise dynamic thresholding (mean similarity per concept)
        tau = S.mean(dim=0) # [Total_Concepts]
        
        # Keep patches that exceed the mean similarity for AT LEAST ONE concept
        valid_mask = (S > tau.unsqueeze(0)).any(dim=1) # [N]
        
        # Fallback safety: If nothing passes, keep everything
        if not valid_mask.any():
            valid_mask = torch.ones_like(valid_mask)
            
        filtered_features = features[valid_mask] # [K, 512]
        
        # --- STAGE 2: VISUAL AGGREGATION & ADAPTATION ---
        embedding, w_filtered = self.aggregator(filtered_features)
        
        res = self.adapter(embedding)
        adapted_emb = self.ratio * res + (1 - self.ratio) * embedding
        adapted_emb = F.normalize(adapted_emb, p=2, dim=-1)
        
        # --- STAGE 3: CLASSIFICATION ---
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (adapted_emb @ prototype_norm.t()) * clamped_scale
        
        # --- XAI MAPPING ---
        # Map the K filtered weights back to the original N dimensions
        w_full = torch.zeros((N, 1), device=features.device, dtype=features.dtype)
        w_full[valid_mask] = w_filtered
        
        return logits, adapted_emb, w_full
        
class ZSMILModel(nn.Module):
    def __init__(self, text_embeddings, abmil_d=64):
        super().__init__()
        self.c_in = 512
        self.aggregator = ABMIL(L=self.c_in, D=abmil_d)
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))

    def forward(self, features):
        embedding, w = self.aggregator(features)
        
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (embedding_norm @ prototype_norm.t()) * clamped_scale
        
        return logits, embedding_norm, w


class TaskResModel(nn.Module):
    def __init__(self, text_embeddings, alpha=0.25, abmil_d=64):
        super().__init__()
        self.c_in = 512
        self.alpha = alpha
        self.aggregator = ABMIL(L=self.c_in, D=abmil_d)
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        
        self.task_res = nn.Parameter(torch.zeros_like(self.text_embeddings))

    def forward(self, features):
        embedding, w = self.aggregator(features)
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        
        adapted_prototype = self.text_embeddings + self.alpha * self.task_res
        prototype_norm = F.normalize(adapted_prototype, p=2, dim=-1)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (embedding_norm @ prototype_norm.t()) * clamped_scale
        
        return logits, embedding_norm, w

def generate_tissue_mask(slide, thumb_factor=32):
    """Global Otsu & Contour Filtering"""
    Wl, Hl = slide.level_dimensions[LEVEL]
    thumb = slide.get_thumbnail((Wl // thumb_factor, Hl // thumb_factor))
    thumb_arr = np.array(thumb.convert("RGB"))

    # 1. Convert to HSV and isolate the Saturation channel
    hsv = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]

    # 2. Gaussian Blur to smooth out noise
    blurred = cv2.GaussianBlur(S, (7, 7), 0)

    # 3. Global Otsu Thresholding (Dynamic separation of tissue vs glass)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Find the structural contours of the thresholded shapes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filter by Area: Keep ONLY massive tissue chunks. 
    # Any contour smaller than 400 thumbnail pixels (crosses, smudges, dust) is destroyed.
    min_area = 400 
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # 6. Create the Macro Mask (General boundaries)
    contour_mask = np.zeros_like(binary, dtype=np.uint8)
    cv2.drawContours(contour_mask, valid_contours, -1, 1, thickness=cv2.FILLED)

    # This physically shrinks the valid boundary inward
    kernel = np.ones((5, 5), np.uint8) 
    eroded_macro_mask = cv2.erode(contour_mask, kernel, iterations=4)

    # 7. Create the Micro Mask (Strict pixel-level color checking for tears/holes)
    S_float = S.astype(np.float32) / 255.0
    V_float = hsv[:, :, 2].astype(np.float32) / 255.0
    is_actual_tissue = (S_float > 0.10) & (V_float > 0.05)

    # 8. Intersection: Must be strictly inside the ERODED boundary AND be physical tissue
    final_mask = (eroded_macro_mask == 1) & is_actual_tissue

    return final_mask, thumb_arr, valid_contours
    
# ==========================================
# PREPARATION & EXTRACTION FUNCTIONS
# ==========================================
def parse_subject_id(filename):
    m = re.search(r"Coh[a-zA-Z0-9]+_(\d+)_", filename)
    return m.group(1) if m else "UNKNOWN"
    
def generate_slide_preview(slide, slide_name, subject_id, thumb_factor=32):
    preview_path = PREVIEWS_DIR / f"{subject_id}_bg_removed_preview.png"
    if preview_path.exists():
        return
        
    mask, thumb_arr, valid_contours = generate_tissue_mask(slide, thumb_factor)
    
    # Create the filtered image by turning everything outside the mask pure white
    processed_thumb = thumb_arr.copy()
    processed_thumb[~mask] = [255, 255, 255] 
    
    # Draw a thin green line around the valid tissue on the original for visualization
    cv2.drawContours(thumb_arr, valid_contours, -1, (0, 255, 0), 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    axes[0].imshow(thumb_arr)
    axes[0].set_title(f"Original Slide (Green = Valid Tissue): {slide_name}", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(processed_thumb)
    axes[1].set_title(f"Otsu & Contour Filtered", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(preview_path, bbox_inches='tight')
    plt.close()
   
def save_final_ml_footprint(slide, slide_name, subject_id, coords, thumb_factor=32):
    """Fills ALL removed parts (glass, holes, blurry, edges) in solid green."""
    footprint_path = PREVIEWS_DIR / f"{subject_id}_final_ml_footprint.png"
    
    # 1. Start with the original slide thumbnail
    Wl, Hl = slide.level_dimensions[LEVEL]
    thumb = slide.get_thumbnail((Wl // thumb_factor, Hl // thumb_factor))
    thumb_arr = np.array(thumb.convert("RGB"))
    
    # ---Calculate the percentage of the slide retained ---
    total_grid_patches = (Wl // PATCH_SIZE) * (Hl // PATCH_SIZE)
    pct_retained = (len(coords) / total_grid_patches) * 100 if total_grid_patches > 0 else 0
    
    # 2. Create a clean white canvas
    h, w, c = thumb_arr.shape
    canvas = np.ones((h, w, c), dtype=np.uint8) * 255
    p_size = PATCH_SIZE // thumb_factor
    
    # 3. Only copy valid tissue patches from original thumbnail onto the canvas
    for (x, y) in coords:
        tx, ty = int(x // thumb_factor), int(y // thumb_factor)
        end_ty = min(ty + p_size, h)
        end_tx = min(tx + p_size, w)
        patch_region = thumb_arr[ty:end_ty, tx:end_tx]
        canvas[ty:end_ty, tx:end_tx] = patch_region
        
    # 4. Fill the remaining white areas (everything not copied) in pure green
    mask = (canvas == [255, 255, 255]).all(axis=-1)
    canvas[mask] = [0, 255, 0] 
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(canvas)
    
    ax.set_title(f"ML Tissue Footprint: {slide_name}\n({pct_retained:.1f}% of Slide Patches Retained)", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(footprint_path, bbox_inches='tight')
    plt.close() 

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
    
    # Grab the SOTA mask
    mask, _, _ = generate_tissue_mask(slide, thumb_factor)
    
    Wl, Hl = slide.level_dimensions[LEVEL]
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
            
            # 1. Must be at least 75% solid tissue on the mask
            if patch_mask.size == 0 or float(patch_mask.mean()) < 0.75:
                continue 
                
            # 2. Extract the patch to check for blur and micro-tears
            patch_img = np.array(slide.read_region((x0, y0), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB"))
            
            # 3. Hole Check (Pixel-level):
            # If the patch itself has too much pure white/light gray, drop it.
            gray_patch = cv2.cvtColor(patch_img, cv2.COLOR_RGB2GRAY)
            white_pixels = np.sum(gray_patch > 220) # 220 out of 255 is very bright/empty
            total_pixels = PATCH_SIZE * PATCH_SIZE
            if (white_pixels / total_pixels) > 0.25: # If more than 25% is pure white, skip it
                continue
                
            # 4. Strict Localized Blur Detection:
            # Split patch into 4 quadrants. If ANY corner is blurry, reject the whole patch.
            half = PATCH_SIZE // 2
            focus_scores = [
                cv2.Laplacian(gray_patch[:half, :half], cv2.CV_64F).var(), # Top-Left
                cv2.Laplacian(gray_patch[:half, half:], cv2.CV_64F).var(), # Top-Right
                cv2.Laplacian(gray_patch[half:, :half], cv2.CV_64F).var(), # Bottom-Left
                cv2.Laplacian(gray_patch[half:, half:], cv2.CV_64F).var()  # Bottom-Right
            ]
            
            if min(focus_scores) < 250.0: # If the blurriest corner is below 250, drop it
                continue
                
            kept_coords.append(torch.tensor([x0, y0]))
    
    if kept_coords:
        save_final_ml_footprint(slide, slide_name, subject_id, kept_coords, thumb_factor)
        
        torch.save({
            "coords": torch.stack(kept_coords),
            "subject_id": subject_id,
            "slide_name": slide_name
        }, out_pt)
    
    slide.close()

class WSIPatchDataset(Dataset):
    def __init__(self, slide_path, coords, patch_size=224, level=0):
        self.slide_path = str(slide_path)
        self.coords = coords
        self.patch_size = patch_size
        self.level = level
        self.slide = None # Lazy loading for multiprocessing safety

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # OpenSlide objects cannot be shared across multiple CPU workers.
        # Initializing it here ensures each worker gets its own instance.
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.slide_path)
            
        x, y = self.coords[idx]
        
        # Read patch directly from the WSI
        patch = np.array(self.slide.read_region((x.item(), y.item()), self.level, (self.patch_size, self.patch_size)).convert("RGB"))
        
        # Convert to tensor and normalize exactly as CONCH expects
        batch_ts = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        
        pixel_values = (batch_ts - mean) / std
        return pixel_values

def run_feature_extraction(device, valid_subjects, subject_meta):
    logging.info("Loading CONCH model for text prototype generation...")
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    conch_checkpoint = PROJECT_ROOT / "models" / "CONCH" / "pytorch_model.bin"
    model, processor = create_model_from_pretrained('conch_ViT-B-16', str(conch_checkpoint))
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer()

    TEMPLATES = ["An H&E stained liver patch showing {}.", "Microscopic view of {}.", "Liver histology demonstrating {}.", "Pathology slide revealing {}."]
    
    CONCEPTS = {"Non-Drinking": [
                    "orderly cords of normal hepatocytes", 
                    "uniform hepatocytes with granular eosinophilic cytoplasm", 
                    "radiating hepatic sinusoids lined by endothelial cells", 
                    "intact portal tracts containing bile ducts and vessels", 
                    "normal central vein architecture"
                ],
                "Drinking": [
                    "macrovesicular steatosis", 
                    "microvesicular lipid accumulation", 
                    "hepatocyte ballooning degeneration", 
                    "neutrophilic lobular inflammation", 
                    "perisinusoidal fibrosis"
                ]}   
                
    text_features_list = []
    concept_dict = {}
    
    pure_concept_dict = {}
    with torch.no_grad():
        for cls in ["Non-Drinking", "Drinking"]:
            for concept in CONCEPTS[cls]:
                if concept not in pure_concept_dict:
                    inputs = tokenizer([concept], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                    outputs = model.encode_text(inputs["input_ids"].to(device))
                    feat = F.normalize(outputs, p=2, dim=-1).squeeze(0).cpu().numpy()
                    pure_concept_dict[concept] = feat
                    
    with torch.no_grad():
        for cls in ["Non-Drinking", "Drinking"]:
            cls_feats = []
            for concept in CONCEPTS[cls]:
                for t in TEMPLATES:
                    prompt = t.format(concept)
                    inputs = tokenizer([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                    outputs = model.encode_text(inputs["input_ids"].to(device))
                    feat = F.normalize(outputs, p=2, dim=-1).squeeze(0).cpu().numpy()
                    
                    concept_dict[prompt] = feat
                    cls_feats.append(feat)
                    
            cls_prototype = np.mean(cls_feats, axis=0)
            text_features_list.append(cls_prototype / np.linalg.norm(cls_prototype))
            
    np.save(FEATURES_DIR / "text_prototypes.npy", np.stack(text_features_list))
    torch.save(concept_dict, FEATURES_DIR / "concept_embeddings.pt")
    torch.save(pure_concept_dict, FEATURES_DIR / "pure_concept_embeddings.pt")
    
    extracted_all = all((FEATURES_DIR / f"{s}_features.pt").exists() for s in valid_subjects)
    if extracted_all:
        logging.info("All selected image features already extracted. Proceeding to training.")
        return

    raw_wsis_dir = PROJECT_ROOT / "dataset" / "raw_wsis"
    pt_files = [PATCHES_DIR / f"{s}_patches.pt" for s in valid_subjects if (PATCHES_DIR / f"{s}_patches.pt").exists()]
    
    for pt_path in tqdm(pt_files, desc="Extracting Image Features"):
        feature_out_path = FEATURES_DIR / pt_path.name.replace("_patches.pt", "_features.pt")
        if feature_out_path.exists():
            continue
            
        # Load the coordinates (very fast, near zero RAM)
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        coords = data["coords"]
        subj_id = data["subject_id"]
        slide_filename = data["slide_name"]
        
        slide_path = raw_wsis_dir / slide_filename
        
        # Create the Dataset and DataLoader
        dataset = WSIPatchDataset(slide_path=slide_path, coords=coords, patch_size=PATCH_SIZE, level=LEVEL)
        # num_workers=4 will pull patches from the hard drive in parallel
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        
        slide_feats = []
        with torch.no_grad():
            for pixel_values in dataloader:
                pixel_values = pixel_values.to(device)
                
                with torch.autocast(device_type=device.type):
                    feats = model.encode_image(pixel_values)
                
                # Move to CPU immediately to free up GPU memory
                slide_feats.append(F.normalize(feats, p=2, dim=-1).cpu().float().numpy())
                
        # Save the final lightweight 512-d features
        torch.save({
            "features": torch.from_numpy(np.concatenate(slide_feats, axis=0)),
            "coords": coords, # Carry the coordinates over for the xAI heatmaps
            "subject_id": subj_id,
            "binary_category": "Non-Drinking" if subject_meta[subj_id]["category"] == "ND" else "Drinking"
        }, feature_out_path)
        
# ==========================================
# DYNAMIC DATASET, SPLIT & TRAINING LOGIC
# ==========================================
def load_or_create_splits(json_path, shots, nd_pool, d_pool, subject_meta, iterations=5):
    """
    Creates dataset splits using a stratified round-robin sampling strategy to ensure
    the 'Drinking' sub-classes (LD, BD, HD, VHD) are uniformly distributed across splits.
    """
    if os.path.exists(json_path):
        logging.info(f"Loading existing dataset splits from {json_path}")
        with open(json_path, 'r') as f:
            return json.load(f)

    logging.info(f"Creating new dataset splits (Shots: {shots}, Iterations: {iterations}) -> {json_path}")
    splits = []
    used_train_nd, used_train_d = set(), set()
    val_shots = min(shots, 4)
    
    def stratified_sample(avail_ids, num_samples):
        # Group available IDs by their exact sub-category (LD, BD, HD, VHD)
        groups = {}
        for sid in avail_ids:
            cat = subject_meta[sid]["category"]
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(sid)

        # Shuffle each group individually to ensure randomness inside categories
        for cat in groups:
            random.shuffle(groups[cat])

        selected = []
        active_cats = list(groups.keys())
        random.shuffle(active_cats) # Randomize starting category to prevent bias
        
        # Round-robin selection to maximize distribution
        while len(selected) < num_samples and active_cats:
            for cat in list(active_cats):
                if len(selected) >= num_samples:
                    break
                selected.append(groups[cat].pop())
                if not groups[cat]:
                    active_cats.remove(cat)
                    
        return selected

    for i in range(iterations):
        avail_nd = list(set(nd_pool) - used_train_nd)
        if len(avail_nd) < shots:
            used_train_nd = set()
            avail_nd = list(nd_pool)
            
        train_nd = random.sample(avail_nd, shots)
        used_train_nd.update(train_nd)

        avail_d = list(set(d_pool) - used_train_d)
        if len(avail_d) < shots:
            used_train_d = set()
            avail_d = list(d_pool)
            
        # Stratified sampling for train (Drinking)
        train_d = stratified_sample(avail_d, shots)
        used_train_d.update(train_d)
        
        avail_val_nd = list(set(nd_pool) - set(train_nd))
        val_nd = random.sample(avail_val_nd, val_shots)
        
        avail_val_d = list(set(d_pool) - set(train_d))
        # Stratified sampling for val (Drinking)
        val_d = stratified_sample(avail_val_d, val_shots)
        
        test_nd = list(set(nd_pool) - set(train_nd) - set(val_nd))
        test_d = list(set(d_pool) - set(train_d) - set(val_d))
        
        splits.append({
            "iteration": i + 1,
            "train_nd": train_nd,
            "train_d": train_d,
            "val_nd": val_nd,
            "val_d": val_d,
            "test_nd": test_nd,
            "test_d": test_d,
            "test_ids": test_nd + test_d
        })

    with open(json_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    return splits

class DynamicWSIDataset(Dataset):
    def __init__(self, subject_ids: List[str]):
        self.subject_ids = subject_ids
        self.label_map = {"Non-Drinking": 0, "Drinking": 1}

    def __len__(self): return len(self.subject_ids)

    def __getitem__(self, idx):
        data = torch.load(FEATURES_DIR / f"{self.subject_ids[idx]}_features.pt", map_location="cpu", weights_only=False)
        return data["features"], self.label_map[data["binary_category"]], self.subject_ids[idx]

def evaluate(model, loader, device, criterion):
    model.eval()
    preds, targets = [], []
    subject_preds = {}
    subject_probs = {} 
    run_loss = 0.0
    with torch.no_grad():
        for feats, label, subj_ids in loader:
            label = label.to(device)
            with torch.autocast(device_type=device.type):
                logits, _, _ = model(feats.to(device).squeeze(0))
                loss = criterion(logits, label)
            run_loss += loss.item()
            
            probs = logits.softmax(dim=-1)
            prob_drinking = probs[0, 1].item()
            
            pred = 1 if prob_drinking >= 0.5 else 0
            preds.append(pred)
            targets.append(label.item())
            
            subj_id = subj_ids[0]
            subject_preds[subj_id] = pred
            subject_probs[subj_id] = round(prob_drinking, 4)
            
    avg_loss = run_loss / len(loader.dataset)
    bacc = sklearn.metrics.balanced_accuracy_score(targets, preds)
    return avg_loss, bacc, subject_preds, subject_probs
    
def train_and_eval_fold(train_ids, val_ids, test_ids, text_prototypes, device, args):
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    train_loader = DataLoader(DynamicWSIDataset(train_ids), batch_size=1, shuffle=True)
    val_loader   = DataLoader(DynamicWSIDataset(val_ids), batch_size=1, shuffle=False)
    test_loader  = DataLoader(DynamicWSIDataset(test_ids), batch_size=1, shuffle=False)
    
    if args.model == "abmil_adapter":
        model = ABMILAdapterModel(text_embeddings=text_prototypes, reduction=args.reduction, abmil_d=args.abmil_d).to(device)
    elif args.model == "zsmil":
        model = ZSMILModel(text_embeddings=text_prototypes, abmil_d=args.abmil_d).to(device)
    elif args.model == "taskres":
        model = TaskResModel(text_embeddings=text_prototypes, alpha=0.25, abmil_d=args.abmil_d).to(device)
    elif args.model == "tgdf_mil":
        pure_dict = torch.load(FEATURES_DIR / "pure_concept_embeddings.pt", map_location="cpu", weights_only=False)
        model = TGDFMILModel(pure_concept_embeddings_dict=pure_dict, text_embeddings=text_prototypes, reduction=args.reduction, abmil_d=args.abmil_d).to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    warmup_epochs = 20
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    
    best_val_loss = float('inf') 
    best_val_bacc = 0.0
    best_weights = None
    patience_ctr = 0
    actual_accum_steps = max(1, min(args.accum_steps, len(train_loader)))
    stopped_epoch = args.epochs
    
    history = {'train_loss': [], 'val_loss': [], 'train_bacc': [], 'val_bacc': []}
    
    pbar = tqdm(range(args.epochs), desc="Training Epochs", leave=False, bar_format='{l_bar}{bar:20}{r_bar}')
    remainder = len(train_loader) 
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        skip_lr_sched = False
        
        for i, (feats, label, _) in enumerate(train_loader):
        
            is_last_chunk = (len(train_loader) - i) <= remainder
            current_divisor = remainder if (is_last_chunk and remainder > 0) else actual_accum_steps
        
            if scaler:
                with torch.autocast(device_type=device.type):
                    logits, _, _ = model(feats.to(device).squeeze(0))
                    loss = criterion(logits, label.to(device)) / actual_accum_steps
                    loss = criterion(logits, label.to(device)) / current_divisor
                scaler.scale(loss).backward()
            else:
                logits, _, _ = model(feats.to(device).squeeze(0))
                loss = criterion(logits, label.to(device)) / actual_accum_steps
                loss.backward()
            
            if (i + 1) % actual_accum_steps == 0 or (i + 1) == len(train_loader):
                if scaler:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    if scale_before > scale_after:
                        skip_lr_sched = True
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
        if not skip_lr_sched:
            scheduler.step()
        
        train_loss, train_bacc, _, _ = evaluate(model, train_loader, device, criterion)
        val_loss, val_bacc, _, _ = evaluate(model, val_loader, device, criterion)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_bacc'].append(train_bacc)
        history['val_bacc'].append(val_bacc)
        
        pbar.set_postfix({'V_Loss': f"{val_loss:.4f}", 'V_Bacc': f"{val_bacc:.4f}"})
        
        if val_loss < best_val_loss:
            patience_ctr = 0
            best_val_loss = val_loss
            best_val_bacc = val_bacc
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE: 
                stopped_epoch = epoch + 1
                break
            
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    
    _, _, val_preds, val_probs = evaluate(model, val_loader, device, criterion)
    test_loss, test_bacc, test_preds, test_probs = evaluate(model, test_loader, device, criterion)
    
    eval_preds = {**val_preds, **test_preds}
    eval_probs = {**val_probs, **test_probs} 
    eval_ids = val_ids + test_ids
    
    return best_val_bacc, test_bacc, stopped_epoch, eval_probs, eval_preds, eval_ids, best_weights, history

def plot_and_save_curves(history, save_dir, iter_idx, seed_idx):
    epochs = range(1, len(history['train_loss']) + 1)
    best_epoch = np.argmin(history['val_loss']) + 1
    
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='orange', linewidth=2, linestyle='--')
    plt.axvline(x=best_epoch, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.title(f"Loss Curve - Iteration {iter_idx} (Seed {seed_idx})", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Cross-Entropy Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(save_dir / f"loss_curve_iter{iter_idx}_seed{seed_idx}.png")
    plt.close()
    
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, history['train_bacc'], label='Train Balanced Accuracy', color='green', linewidth=2)
    plt.plot(epochs, history['val_bacc'], label='Validation Balanced Accuracy', color='red', linewidth=2, linestyle='--')
    plt.axvline(x=best_epoch, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.title(f"Accuracy Curve - Iteration {iter_idx} (Seed {seed_idx})", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Balanced Accuracy", fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(save_dir / f"acc_curve_iter{iter_idx}_seed{seed_idx}.png")
    plt.close()

# ==========================================
# MAIN EXECUTION & HYBRID xAI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["abmil_adapter", "zsmil", "taskres", "tgdf_mil"], default="tgdf_mil")
    parser.add_argument('--epochs', type=int, default=200, help="Max training epochs.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--reduction', type=int, default=2, help="Adapter bottleneck reduction factor.")
    parser.add_argument('--abmil_d', type=int, default=64, help="ABMIL projection dimension (D). Controls the bottleneck of the attention mechanism.")
    parser.add_argument('--shots', type=int, default=1, help="Number of shots (samples per class) for training.")
    parser.add_argument('--iterations', type=int, default=5, help="Number of random splits/iterations.")
    parser.add_argument('--dataset_pct', type=float, default=1.0, help="Percentage of total dataset to use (e.g., 0.5 for 50%).")
    parser.add_argument('--seeds', type=int, default=1, help="Number of random seeds per fold.") 
    args = parser.parse_args()
    
    args.accum_steps = args.shots * 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- DYNAMIC 37-SUBJECT DATASET PARSING & SUBSET SELECTION ---
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
        logging.info(f"Subsampling Dataset to {args.dataset_pct*100}%: {nd_keep} ND and {d_keep} D subjects selected.")

    valid_subjects = set(ND_POOL + D_POOL)
    selected_svs_filenames = [f"Coh{SUBJECT_META[s]['cohort']}_{s}_Nx_Liver_HE.svs" for s in valid_subjects]
    raw_wsis = {parse_subject_id(f): PROJECT_ROOT / "dataset" / "raw_wsis" / f for f in selected_svs_filenames}

    SHARED_SPLITS_DIR = STANDALONE_DIR / "shared_splits"
    SHARED_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON_PATH = SHARED_SPLITS_DIR / f"dataset_splits_{args.shots}shot_{args.iterations}iter_{int(args.dataset_pct*100)}pct.json"
    
    if args.model == "abmil_adapter":
        model_str = f"{args.model}_r{args.reduction}_d{args.abmil_d}"
    else:
        model_str = f"{args.model}_d{args.abmil_d}"
        
    DYNAMIC_RESULTS_DIR = STANDALONE_DIR / f"{model_str}_{args.shots}shot_{args.iterations}iter"
    CURVES_DIR = DYNAMIC_RESULTS_DIR / "training_curves"
    PROBS_DIR = DYNAMIC_RESULTS_DIR / "test_probabilities" 
    
    DYNAMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    PROBS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "abmil_adapter": dummy = ABMILAdapterModel(text_embeddings=np.zeros((2, 512)), reduction=args.reduction, abmil_d=args.abmil_d)
    elif args.model == "zsmil": dummy = ZSMILModel(text_embeddings=np.zeros((2, 512)), abmil_d=args.abmil_d)
    elif args.model == "taskres": dummy = TaskResModel(text_embeddings=np.zeros((2, 512)), alpha=0.25, abmil_d=args.abmil_d)
    elif args.model == "tgdf_mil": dummy = TGDFMILModel(pure_concept_embeddings_dict={f"c{i}": np.zeros(512) for i in range(10)}, text_embeddings=np.zeros((2, 512)), abmil_d=args.abmil_d)
    logging.info(f"Using {args.model.upper()} | Trainable Params: {count_trainable_parameters(dummy):,} | Shots: {args.shots} | Epochs: {args.epochs}")
    del dummy
    
    logging.info("Checking/Extracting Patches & Generating Previews...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(tqdm(ex.map(_extract_patches_worker, list(raw_wsis.values())), total=len(raw_wsis)))
        
    run_feature_extraction(device, valid_subjects, SUBJECT_META)
    
    text_prototypes = np.load(FEATURES_DIR / "text_prototypes.npy")
    val_baccs, test_baccs, combo_report_logs, all_epochs, xai_infos = [], [], [], [], []
    
    dataset_splits = load_or_create_splits(SPLITS_JSON_PATH, args.shots, ND_POOL, D_POOL, SUBJECT_META, args.iterations)
    
    logging.info(f"Starting Few-Shot Evaluation ({len(dataset_splits)} Iterations) with {args.seeds} Seeds per Fold")
    
    for fold_idx, split_info in enumerate(dataset_splits, start=1):
        fold_val_bacc, fold_test_bacc, fold_epochs = [], [], []
        train_ids = split_info["train_nd"] + split_info["train_d"]
        val_ids = split_info["val_nd"] + split_info["val_d"]
        test_ids = split_info["test_ids"]
        
        for n_seed in range(args.seeds):
            set_seed(n_seed)

            val_bacc, test_bacc, stopped_epoch, eval_probs, eval_preds, eval_ids, b_weights, history = train_and_eval_fold(
                train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
                text_prototypes=text_prototypes, device=device, args=args
            )

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
                
                t_class = "Drinking" if true_label == 1 else "Non-Drinking"
                p_class = "Drinking" if pred_label == 1 else "Non-Drinking"
                correct = "Yes" if true_label == pred_label else "No"
                
                txt_lines.append(f"{subj:<15} | {t_class:<15} | {p_class:<15} | {correct:<8} | {prob:<15.4f} | {conf_str}")

            txt_lines.append("============================================================")
            
            prob_file = PROBS_DIR / f"iter{fold_idx:02d}_seed{n_seed}.txt"
            with open(prob_file, "w") as f:
                f.write("\n".join(txt_lines))

            plot_and_save_curves(history, CURVES_DIR, fold_idx, n_seed)
            fold_val_bacc.append(val_bacc)
            fold_test_bacc.append(test_bacc)
            fold_epochs.append(stopped_epoch)
            
            if fold_idx == 1 and n_seed == 0:
                for subj_id in eval_ids:
                    if subj_id in D_POOL and eval_preds.get(subj_id) == 1:
                        xai_infos.append({
                            "wsi_id": subj_id, 
                            "combo_idx": fold_idx, 
                            "seed_idx": n_seed, 
                            "weights": b_weights
                        })
        
        avg_val_bacc, std_val_bacc = np.mean(fold_val_bacc), np.std(fold_val_bacc)
        avg_test_bacc, std_test_bacc = np.mean(fold_test_bacc), np.std(fold_test_bacc)
        avg_epochs, std_epochs = np.mean(fold_epochs), np.std(fold_epochs)
        
        val_baccs.append(avg_val_bacc)
        test_baccs.append(avg_test_bacc)
        all_epochs.append(avg_epochs)
        
        log_str = (f"Iter {fold_idx:02d}/{args.iterations} | Train:[{','.join(train_ids)}] "
                   f"Val:[{','.join(val_ids)}] Test:[{len(test_ids)} subjects] | "
                   f"Epochs: {avg_epochs:.1f}±{std_epochs:.1f} | "
                   f"Val BACC: {avg_val_bacc:.4f}±{std_val_bacc:.4f} | "
                   f"Test BACC: {avg_test_bacc:.4f}±{std_test_bacc:.4f}")
        logging.info(log_str)
        combo_report_logs.append(f"Iter {fold_idx:02d}: Train:[{','.join(train_ids)}] | Test Subjects: [{len(test_ids)}] -> Test BACC: {avg_test_bacc:.4f} ± {std_test_bacc:.4f}")
            
    summary = (
        f"\n{'='*80}\nFEW-SHOT EVALUATION SUMMARY (Model: {args.model.upper()} - {args.shots} Shot - {int(args.dataset_pct*100)}% Data)\n{'='*80}\n"
        f"Average Epochs: {np.mean(all_epochs):.1f} ± {np.std(all_epochs):.1f}\n{'='*80}\n"
        f"INDIVIDUAL COMBO DETAILS:\n" + "\n".join(combo_report_logs) + f"\n{'='*80}\n"
        f"OVERALL METRICS (Mean of {args.iterations} Iterations):\n"
        f"Validation BACC: {np.mean(val_baccs):.4f} ± {np.std(val_baccs):.4f}\n"
        f"Test BACC:        {np.mean(test_baccs):.4f} ± {np.std(test_baccs):.4f}\n{'='*80}"
    )
    print(summary)
    with open(DYNAMIC_RESULTS_DIR / f"{args.shots}_shot_metrics.txt", "w") as f: f.write(summary)

    if xai_infos:
        if args.model in ["abmil_adapter", "zsmil", "taskres", "tgdf_mil"]: xai_mode = "Attention-Based"
        else: xai_mode = "Gradient-Based"
        concept_dict = torch.load(FEATURES_DIR / "concept_embeddings.pt", map_location="cpu", weights_only=False)
        DRINKING_CONCEPTS = ["macrovesicular steatosis", "microvesicular lipid accumulation", "hepatocyte ballooning degeneration", "neutrophilic lobular inflammation", "perisinusoidal fibrosis"]
        drinking_sentences = [sent for sent in concept_dict.keys() if any(c in sent for c in DRINKING_CONCEPTS)]

        for xai_info in xai_infos:
            wsi_id = xai_info['wsi_id']
            iter_idx = xai_info['combo_idx']
            seed_idx = xai_info['seed_idx']
            
            XAI_OUT_DIR = DYNAMIC_RESULTS_DIR / f"xAI_Iter{iter_idx:02d}_Seed{seed_idx}"
            XAI_OUT_DIR.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Running {xai_mode} xAI Analysis on WSI {wsi_id}... Saving to {XAI_OUT_DIR.name}")
            
            if args.model == "abmil_adapter":
                xai_model = ABMILAdapterModel(text_embeddings=text_prototypes, reduction=args.reduction, abmil_d=args.abmil_d).to(device)
            elif args.model == "zsmil":
                xai_model = ZSMILModel(text_embeddings=text_prototypes, abmil_d=args.abmil_d).to(device)
            elif args.model == "taskres":
                xai_model = TaskResModel(text_embeddings=text_prototypes, alpha=0.25, abmil_d=args.abmil_d).to(device)
            elif args.model == "tgdf_mil":
                pure_dict = torch.load(FEATURES_DIR / "pure_concept_embeddings.pt", map_location="cpu", weights_only=False)
                xai_model = TGDFMILModel(pure_concept_embeddings_dict=pure_dict, text_embeddings=text_prototypes, reduction=args.reduction, abmil_d=args.abmil_d).to(device)
                
            xai_model.load_state_dict({k: v.to(device) for k, v in xai_info["weights"].items()})
            xai_model.eval()

            feat_data = torch.load(FEATURES_DIR / f"{wsi_id}_features.pt", map_location="cpu", weights_only=False)
            coords = feat_data["coords"].numpy()
            features = feat_data["features"].to(device)
            mass_msg = ""
            
            if args.model in ["abmil_adapter", "zsmil", "taskres", "tgdf_mil"]:
                with torch.no_grad():
                    with torch.autocast(device_type=device.type):
                        logits, slide_emb, w = xai_model(features)
                    importance_scores = w.squeeze().cpu().numpy()
                
                sorted_scores = np.sort(importance_scores)[::-1]
                cum_scores = np.cumsum(sorted_scores)
                patches_50 = int(np.argmax(cum_scores >= 0.5) + 1) 
                total_p = len(importance_scores)
                pct_of_tissue = (patches_50 / total_p) * 100
                
                mass_msg = (f"\n--- Attention Mass Diagnostics ---\n"
                            f"To reach 50% of the total decision weight, the model relied on:\n"
                            f"{patches_50} out of {total_p} total patches ({pct_of_tissue:.2f}% of the tissue).\n"
                            f"A lower percentage indicates localization.\n")
            
            else:
                steps = 50
                baseline = torch.zeros_like(features)
                avg_grads = torch.zeros_like(features)
                
                for i in range(1, steps + 1):
                    alpha = i / steps
                    step_feat = (baseline + alpha * (features - baseline)).detach().requires_grad_(True)
                    with torch.autocast(device_type=device.type):
                        logits, slide_emb_step, _ = xai_model(step_feat)
                        drinking_score = logits[0, 1]
                    
                    xai_model.zero_grad()
                    drinking_score.backward()
                    avg_grads += step_feat.grad / steps
                
                ig_attributions = (features - baseline) * avg_grads
                importance_scores = torch.linalg.norm(ig_attributions, dim=1).cpu().numpy()
                with torch.no_grad(), torch.autocast(device_type=device.type):
                    _, slide_emb, _ = xai_model(features)
                
                mass_msg = "\n--- Attention Mass Diagnostics ---\nN/A.\n"
                
            # --- START HEATMAP GENERATION ---
            slide_path = raw_wsis[wsi_id]
            slide = openslide.OpenSlide(str(slide_path))
            thumb_factor = 32
            Wl, Hl = slide.level_dimensions[LEVEL]
            thumb = slide.get_thumbnail((Wl // thumb_factor, Hl // thumb_factor))
            
            # Initialize grids: One for the scores, one boolean mask for the tissue location
            heatmap_grid = np.zeros((Hl // thumb_factor, Wl // thumb_factor), dtype=np.float32)
            tissue_mask = np.zeros((Hl // thumb_factor, Wl // thumb_factor), dtype=bool) 
            p_size = PATCH_SIZE // thumb_factor
            
            # Explicitly mark tissue locations using the known patch coordinates
            for idx, (x, y) in enumerate(coords):
                tx, ty = int(x // thumb_factor), int(y // thumb_factor)
                heatmap_grid[ty:ty+p_size, tx:tx+p_size] = importance_scores[idx]
                tissue_mask[ty:ty+p_size, tx:tx+p_size] = True 
                
            # Smooth the grid
            heatmap_smoothed = scipy.ndimage.gaussian_filter(heatmap_grid, sigma=p_size)
            
            # Calculate Min/Max using the confirmed tissue patches
            if tissue_mask.any():
                tissue_min = heatmap_smoothed[tissue_mask].min()
                tissue_max = heatmap_smoothed[tissue_mask].max()
            else:
                tissue_min, tissue_max = 0.0, 1.0 # Fallback safety
            
            # Normalize based entirely on the tissue range
            heatmap_norm = (heatmap_smoothed - tissue_min) / (tissue_max - tissue_min + 1e-8)
            heatmap_norm = np.clip(heatmap_norm, 0.0, 1.0) 
            
            fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
            ax.imshow(thumb)
            cmap = plt.cm.jet
            overlay = cmap(heatmap_norm)
            
            # Apply alpha to tissue regions. 
            overlay[..., 3] = np.where(tissue_mask, heatmap_norm * 0.6, 0.0)
            
            ax.imshow(overlay)
            ax.set_title(f"{xai_mode} Importance Heatmap: {wsi_id}", fontsize=16)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(XAI_OUT_DIR / f"xai_heatmap_overlay_{wsi_id}.png", bbox_inches='tight')
            plt.close()
            
            # --- START TOP-K PATCH EXTRACTION ---
            top_k = 15
            top_indices = np.argsort(importance_scores)[-top_k:][::-1].copy()
            fig, axes = plt.subplots(3, 5, figsize=(15, 9), dpi=300) 
            axes = axes.flatten()
            
            for i, (ax_, idx) in enumerate(zip(axes, top_indices)):
                x, y = coords[idx]
                patch_img = np.array(slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB"))
                ax_.imshow(patch_img)
                ax_.set_title(f"Rank {i+1}", fontsize=12)
                for spine in ax_.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
                ax_.set_xticks([])
                ax_.set_yticks([])
                
            plt.tight_layout()
            plt.savefig(XAI_OUT_DIR / f"xai_top_patches_{wsi_id}.png", bbox_inches='tight')
            plt.close()

            # --- START PATCH-TO-PURE-CONCEPT MATCHING ---
            top_patch_feats = features[top_indices.copy()].detach().cpu().float()
            top_patch_feats = F.normalize(top_patch_feats, p=2, dim=-1).numpy()
            
            pure_concept_dict = torch.load(FEATURES_DIR / "pure_concept_embeddings.pt", map_location="cpu", weights_only=False)
            concept_phrases = list(pure_concept_dict.keys())
            concept_matrix = np.stack(list(pure_concept_dict.values())) # Shape: (Total_Concepts, 512)
            
            similarity_matrix = top_patch_feats @ concept_matrix.T 
            
            pure_concept_report_path = XAI_OUT_DIR / f"xai_top15_pure_concepts_{wsi_id}.txt"
            with open(pure_concept_report_path, "w") as f:
                f.write("Top 15 Patches - Pure Concept Alignment\n")
                f.write("=======================================\n")
                for i, idx in enumerate(top_indices):
                    patch_sims = similarity_matrix[i]
                    best_concept_idx = np.argmax(patch_sims)
                    best_phrase = concept_phrases[best_concept_idx]
                    best_score = patch_sims[best_concept_idx]
                    
                    f.write(f"Rank {i+1:02d} (Patch Index {idx}):\n")
                    f.write(f"  Coordinates  : X={coords[idx][0]:<6}, Y={coords[idx][1]}\n")
                    f.write(f"  Top Match    : {best_phrase}\n")
                    f.write(f"  Cosine Score : {best_score:.4f}\n\n")
                    
            # --- START REPORT GENERATION ---
            slide_emb_float = slide_emb.detach().cpu().float()
            slide_emb_norm = F.normalize(slide_emb_float, p=2, dim=-1).numpy().squeeze()
            
            sentence_scores = {}
            for sentence in drinking_sentences:
                sentence_scores[sentence] = float(np.dot(slide_emb_norm, concept_dict[sentence]))
            ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

            with open(XAI_OUT_DIR / f"xai_report_{wsi_id}.txt", "w") as f:
                f.write("=====================================================\n")
                f.write(f"      xAI ALIGNMENT & {xai_mode.upper()} REPORT\n")
                f.write("=====================================================\n\n")
                f.write(f"Model Architecture Used : {args.model}\n")
                f.write(f"Evaluated WSI Subject ID: {wsi_id} (True Label: Drinking, Predicted: Drinking)\n")
                f.write(f"Generated from Iteration: {iter_idx} | Seed: {seed_idx}\n")
                
                f.write(mass_msg) 
                
                f.write("\n--- Slide-Level Morphological Profiling ---\n")
                f.write("Cosine Similarity (-1 to +1) of the final WSI embedding against all specific training strings:\n")
                for rank, (sentence, score) in enumerate(ranked_sentences):
                    f.write(f"{rank+1:02d}. Score: {score:+.4f} | {sentence}\n")
                
                f.write("\n--- Patch-Level Saliency (Top 15) ---\n")
                for rank, idx in enumerate(top_indices):
                    f.write(f"Rank {rank+1:02d}:\n")
                    f.write(f"  Coordinates : X={coords[idx][0]:<6}, Y={coords[idx][1]}\n")
                    f.write(f"  Score       : {importance_scores[idx]:.6f}\n")
            
            logging.info(f"xAI Analysis complete for WSI {wsi_id}.")
            
            slide.close()

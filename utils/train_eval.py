"""
train_eval.py
Training, Evaluation, and Dataset Modules.
Provides utilities for dataset handling, stratified splitting, 
and the main training/validation loops for the MIL models.
"""

import os
import json
import random
import logging
from typing import List
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sklearn.metrics
import openslide
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import matplotlib.pyplot as plt

from .mil_models import (
    ABMILAdapterModel, 
    ZSMILModel, 
    TaskResModel, 
    MaxPoolModel, 
    PromptLearningMILModel,
    TOPModel
)

class WSIPatchDataset(Dataset):
    """
    Dataset for loading Whole Slide Image (WSI) patches on the fly.
    Initializes OpenSlide objects per-worker for multiprocessing.
    """
    def __init__(self, slide_path, coords, patch_size=224, level=0):
        self.slide_path = str(slide_path)
        self.coords = coords
        self.patch_size = patch_size
        self.level = level
        self.slide = None 

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.slide_path)
            
        x, y = self.coords[idx]
        
        patch = np.array(
            self.slide.read_region(
                (x.item(), y.item()), 
                self.level, 
                (self.patch_size, self.patch_size)
            ).convert("RGB")
        )
        
        batch_ts = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        
        pixel_values = (batch_ts - mean) / std
        return pixel_values


def load_or_create_splits(json_path, shots, nd_pool, d_pool, subject_meta, iterations=5):
    """
    Loads existing dataset splits or creates new ones using a stratified.
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
        groups = {}
        for sid in avail_ids:
            cat = subject_meta[sid]["category"]
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(sid)

        for cat in groups:
            random.shuffle(groups[cat])

        selected = []
        active_cats = list(groups.keys())
        random.shuffle(active_cats) 
        
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
            
        train_d = stratified_sample(avail_d, shots)
        used_train_d.update(train_d)
        
        avail_val_nd = list(set(nd_pool) - set(train_nd))
        val_nd = random.sample(avail_val_nd, val_shots)
        
        avail_val_d = list(set(d_pool) - set(train_d))
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
    """
    Dataset for dynamically loading pre-extracted WSI features.
    """
    def __init__(self, subject_ids: List[str], features_dir: Path):
        self.subject_ids = subject_ids
        self.features_dir = features_dir
        self.label_map = {
                    "Non-Drinking": 0, 
                    "Drinking": 1,
                    "Healthy Liver": 0, 
                    "Alcoholic Liver Disease": 1
                }
    def __len__(self): 
        return len(self.subject_ids)

    def __getitem__(self, idx):
        data = torch.load(
            self.features_dir / f"{self.subject_ids[idx]}_features.pt", 
            map_location="cpu", 
            weights_only=False
        )
        return data["features"], self.label_map[data["binary_category"]], self.subject_ids[idx]


def evaluate(model, loader, device, criterion):
    """
    Evaluates the model on a given dataloader.
    Returns average loss, balanced accuracy, and predictions.
    """
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
       
    
def train_and_eval_fold(train_ids, val_ids, test_ids, text_prototypes, device, args, features_dir, clip_model=None, tokenizer=None, bag_concepts=None, instance_concepts=None, bag_templates=None, instance_templates=None):
    """
    Executes the training and evaluation loop for a single fold/iteration.
    """
    train_loader = DataLoader(DynamicWSIDataset(train_ids, features_dir), batch_size=1, shuffle=True)
    val_loader   = DataLoader(DynamicWSIDataset(val_ids, features_dir), batch_size=1, shuffle=False)
    test_loader  = DataLoader(DynamicWSIDataset(test_ids, features_dir), batch_size=1, shuffle=False)
    
    if args.model == "abmil_adapter":
        model = ABMILAdapterModel(text_embeddings=text_prototypes, reduction=args.reduction, abmil_d=args.abmil_d, fusion=args.fusion).to(device)
    elif args.model == "zsmil":
        model = ZSMILModel(text_embeddings=text_prototypes, abmil_d=args.abmil_d).to(device)
    elif args.model == "taskres":
        model = TaskResModel(text_embeddings=text_prototypes, alpha=0.25, abmil_d=args.abmil_d, fusion=args.fusion).to(device)
    elif args.model == "max_pool":
        model = MaxPoolModel(text_embeddings=text_prototypes, head_type=args.max_pool_head, reduction=args.reduction, alpha=0.25).to(device)
    elif args.model == "coop":
        model = PromptLearningMILModel(
            clip_model=clip_model, 
            tokenizer=tokenizer, 
            concepts_dict=concepts, 
            templates=templates, 
            n_ctx=args.n_ctx, 
            abmil_d=args.abmil_d,
            fusion=args.fusion,
        ).to(device)
    elif args.model == "top":
        model = TOPModel(
            clip_model=clip_model, 
            tokenizer=tokenizer, 
            bag_concepts_dict=bag_concepts, 
            instance_concepts_dict=instance_concepts, 
            bag_templates=bag_templates, 
            instance_templates=instance_templates,
            n_ctx=args.n_ctx
        ).to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    warmup_epochs = 20
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
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
    remainder = len(train_loader) % actual_accum_steps
    
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
                    loss_ce = criterion(logits, label.to(device))
                    loss = loss_ce / current_divisor

                scaler.scale(loss).backward()
            else:
                logits, _, _ = model(feats.to(device).squeeze(0))
                loss_ce = criterion(logits, label.to(device))
                loss = loss_ce / current_divisor
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
            if patience_ctr >= args.patience: 
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
    """Generates and saves visual training/validation loss and accuracy curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = range(1, len(history['train_loss']) + 1)
    best_epoch = np.argmin(history['val_loss']) + 1
    
    # --- Plot Loss ---
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
    plt.savefig(save_dir / f"loss_curve_iter{iter_idx}_seed{seed_idx}.png")
    plt.close()
    
    # --- Plot Accuracy ---
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
    plt.savefig(save_dir / f"acc_curve_iter{iter_idx}_seed{seed_idx}.png")
    plt.close()    

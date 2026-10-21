"""
train_eval.py
Training, Evaluation, and Dataset Modules.
Provides utilities for dataset handling, stratified splitting, and the main training/validation loops.
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

from .mil_models import BASELINE_MODELS, build_model
from .augmentation import load_augmented_views
from .baselines import fit_baseline
from .contrastive import supervised_contrastive_loss, validate_supcon_options

class WSIPatchDataset(Dataset):
    """
    Dataset for loading Whole Slide Image (WSI) patches on the fly.
    Initializes OpenSlide objects per-worker for multiprocessing.
    """
    def __init__(self, slide_path, coords, patch_size=224, level=0, preprocess=None, augmentation=None):
        if preprocess is None:
            raise ValueError("Pass the image transform returned by CONCH's model loader.")
        self.slide_path = str(slide_path)
        self.coords = coords
        self.patch_size = patch_size
        self.level = level
        self.slide = None 
        self.preprocess = preprocess
        self.augmentation = augmentation

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.slide_path)
            
        x, y = self.coords[idx]
        
        patch = self.slide.read_region(
            (x.item(), y.item()), self.level, (self.patch_size, self.patch_size)
        ).convert("RGB")
        if self.augmentation is not None:
            patch = self.augmentation(patch, idx)
        return self.preprocess(patch)

def validate_splits(splits, shots, nd_pool, d_pool, iterations):
    """Check subject IDs, class counts, and train/validation/test separation."""
    if len(set(nd_pool)) != len(nd_pool) or len(set(d_pool)) != len(d_pool) or set(nd_pool) & set(d_pool):
        raise ValueError("Subject pools must contain unique, disjoint IDs.")
    if len(splits) != iterations:
        raise ValueError("Cached split count differs from requested iterations.")
    pools = {"nd": set(nd_pool), "d": set(d_pool)}
    for index, split in enumerate(splits, 1):
        if split["iteration"] != index:
            raise ValueError("Split iteration IDs must be consecutive.")
        phases = {}
        for phase in ("train", "val", "test"):
            phases[phase] = []
            for cls, pool in pools.items():
                ids = split[f"{phase}_{cls}"]
                if len(ids) != len(set(ids)) or not set(ids) <= pool:
                    raise ValueError(f"Invalid {phase}_{cls} IDs in split {index}.")
                expected = shots if phase == "train" else min(shots, 4)
                if phase != "test" and len(ids) != expected:
                    raise ValueError(f"Wrong {phase}_{cls} size in split {index}.")
                if phase == "test" and not ids:
                    raise ValueError(f"Both test classes must be nonempty in split {index}.")
                phases[phase].extend(ids)
        all_ids = sum(phases.values(), [])
        if len(all_ids) != len(set(all_ids)) or set(all_ids) != pools["nd"] | pools["d"]:
            raise ValueError(f"Subject leakage or missing subjects in split {index}.")
        if split["test_ids"] != phases["test"]:
            raise ValueError(f"test_ids do not match the test partition in split {index}.")


def load_or_create_splits(json_path, shots, nd_pool, d_pool, subject_meta, iterations=5, split_seed=42):
    if shots < 1 or iterations < 1 or min(len(nd_pool), len(d_pool)) <= shots + min(shots, 4):
        raise ValueError("Need positive shots/iterations and at least one test subject per class.")
    if os.path.exists(json_path):
        logging.info(f"Loading existing dataset splits from {json_path}")
        with open(json_path, 'r') as f:
            splits = json.load(f)
        validate_splits(splits, shots, nd_pool, d_pool, iterations)
        return splits

    logging.info(f"Creating new dataset splits (Shots: {shots}, Iterations: {iterations}) -> {json_path}")
    splits = []
    used_train_nd, used_train_d = set(), set()
    val_shots = min(shots, 4)
    rng = random.Random(split_seed)
    
    def stratified_sample(avail_ids, num_samples):
        groups = {}
        for sid in avail_ids:
            cat = subject_meta[sid]["category"]
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(sid)

        for cat in groups:
            rng.shuffle(groups[cat])

        selected = []
        active_cats = list(groups.keys())
        rng.shuffle(active_cats)
        
        while len(selected) < num_samples and active_cats:
            for cat in list(active_cats):
                if len(selected) >= num_samples:
                    break
                selected.append(groups[cat].pop())
                if not groups[cat]:
                    active_cats.remove(cat)
                    
        return selected

    for i in range(iterations):
        avail_nd = sorted(set(nd_pool) - used_train_nd)
        if len(avail_nd) < shots:
            used_train_nd = set()
            avail_nd = sorted(nd_pool)
            
        train_nd = rng.sample(avail_nd, shots)
        used_train_nd.update(train_nd)

        avail_d = sorted(set(d_pool) - used_train_d)
        if len(avail_d) < shots:
            used_train_d = set()
            avail_d = sorted(d_pool)
            
        train_d = stratified_sample(avail_d, shots)
        used_train_d.update(train_d)
        
        avail_val_nd = sorted(set(nd_pool) - set(train_nd))
        val_nd = rng.sample(avail_val_nd, val_shots)
        
        avail_val_d = sorted(set(d_pool) - set(train_d))
        val_d = stratified_sample(avail_val_d, val_shots)
        
        test_nd = sorted(set(nd_pool) - set(train_nd) - set(val_nd))
        test_d = sorted(set(d_pool) - set(train_d) - set(val_d))
        
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

    validate_splits(splits, shots, nd_pool, d_pool, iterations)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    return splits

class DynamicWSIDataset(Dataset):
    def __init__(self, subject_ids: List[str], features_dir: Path, augmentation_dir=None):
        self.subject_ids = subject_ids
        self.features_dir = features_dir
        self.augmentation_dir = Path(augmentation_dir) if augmentation_dir is not None else None
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
        features = data["features"]
        if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != 512 or not torch.isfinite(features).all():
            raise ValueError(f"Invalid feature bag for subject {self.subject_ids[idx]}.")
        if self.augmentation_dir is not None:
            views = load_augmented_views(
                self.augmentation_dir / f"{self.subject_ids[idx]}_aug_features.pt", features, self.subject_ids[idx])
            # Randomly select the original or an augmented feature set for this slide.
            choice = int(torch.randint(len(views) + 1, (1,)).item())
            if choice:
                features = views[choice - 1]
        return features, self.label_map[data["binary_category"]], self.subject_ids[idx]

def evaluate(model, loader, device, criterion):
    """
    Evaluates the model on a given dataloader.
    Returns average loss, BACC, AUC, F1, and predictions.
    """
    model.eval()
    preds, targets, prob_list = [], [], []
    subject_preds = {}
    subject_probs = {} 
    run_loss = 0.0
    
    with torch.no_grad():
        for feats, label, subj_ids in loader:
            label = label.to(device)
            # Evaluate in float32 so close probabilities retain their ROC ranking.
            outputs = model(feats.to(device).squeeze(0))
            logits = outputs[0].float()
            if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                raise ValueError("Expected finite binary logits with shape (1, 2).")
            loss = criterion(logits, label)
            run_loss += loss.item()
            
            probs = logits.softmax(dim=-1)
            prob_drinking = probs[0, 1].item()
            
            pred = 1 if prob_drinking >= 0.5 else 0
            preds.append(pred)
            targets.append(label.item())
            prob_list.append(prob_drinking) 
            
            subj_id = subj_ids[0]
            subject_preds[subj_id] = pred
            subject_probs[subj_id] = prob_drinking
            
    if not targets:
        raise ValueError("Cannot evaluate an empty partition.")
    if len(set(targets)) != 2:
        raise ValueError("ROC AUC requires both classes; do not substitute chance AUC.")
    avg_loss = run_loss / len(loader.dataset)
    bacc = sklearn.metrics.balanced_accuracy_score(targets, preds)
    f1 = sklearn.metrics.f1_score(targets, preds, zero_division=0)
    
    auc = sklearn.metrics.roc_auc_score(targets, prob_list)
        
    return avg_loss, bacc, auc, f1, subject_preds, subject_probs
        
def snapshot_state(model):
    """Clone even on CPU: .cpu() alone shares storage with the live model."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_and_eval_fold(train_ids, val_ids, test_ids, bag_embeddings, instance_embeddings, device, args, features_dir, augmentation_dir=None):
    """
    Executes the training and evaluation loop for a single fold/iteration.
    """
    validate_supcon_options(args)
    supcon_weight = getattr(args, "supcon_weight", 0.0)
    supcon_temperature = getattr(args, "supcon_temperature", 0.1)
    if supcon_weight and (len(set(train_ids)) != len(train_ids) or augmentation_dir is not None):
        raise ValueError("Contrastive training requires distinct training animals and original image views.")
    train_dataset = DynamicWSIDataset(train_ids, features_dir)
    train_loader = DataLoader(DynamicWSIDataset(train_ids, features_dir, augmentation_dir), batch_size=1, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader   = DataLoader(DynamicWSIDataset(val_ids, features_dir), batch_size=1, shuffle=False)
    test_loader  = DataLoader(DynamicWSIDataset(test_ids, features_dir), batch_size=1, shuffle=False)
    
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    model = build_model(args, bag_embeddings).to(device)
    if args.model in BASELINE_MODELS:
        fit_baseline(model, train_dataset, augmentation_dir,
                     regularization_c=getattr(args, "baseline_c", 1.0))
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        train_loss, train_bacc, _, _, _, _ = evaluate(model, train_eval_loader, device, criterion)
        val_loss, val_bacc, _, _, val_preds, val_probs = evaluate(model, val_loader, device, criterion)
        _, test_bacc, test_auc, test_f1, test_preds, test_probs = evaluate(model, test_loader, device, criterion)
        history = {"train_loss": [train_loss], "val_loss": [val_loss],
                   "train_bacc": [train_bacc], "val_bacc": [val_bacc]}
        return (val_bacc, test_bacc, test_auc, test_f1, 1, {**val_probs, **test_probs},
                {**val_preds, **test_preds}, val_ids + test_ids, snapshot_state(model), history)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # Limit the warmup to 20 epochs, leaving at least one cosine-schedule epoch.
    warmup_epochs = min(20, max(0, args.epochs - 1))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6)
    if warmup_epochs:
        warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = cosine_sched

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    
    best_val_loss = float('inf') 
    best_val_bacc = 0.0
    best_weights = None
    patience_ctr = 0
    actual_accum_steps = max(1, min(args.accum_steps, len(train_loader)))
    stopped_epoch = args.epochs
    
    history = {'train_loss': [], 'val_loss': [], 'train_bacc': [], 'val_bacc': []}
    if supcon_weight:
        history.update(objective_ce=[], objective_supcon=[], objective_total=[])
    pbar = tqdm(range(args.epochs), desc="Training Epochs", leave=False, bar_format='{l_bar}{bar:20}{r_bar}')
    remainder = len(train_loader) % actual_accum_steps
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        skip_lr_sched = False
        
        if supcon_weight:
            # Retain all training graphs for the joint animal-level objective.
            embeddings, labels, classification_losses = [], [], []
            for feats, label, _ in train_loader:
                label = label.to(device)
                with torch.autocast(device_type=device.type, enabled=scaler is not None):
                    logits, embedding, _ = model(feats.to(device).squeeze(0))
                    classification_losses.append(criterion(logits, label))
                embeddings.append(embedding)
                labels.append(label)
            loss_ce = torch.stack(classification_losses).mean()
            loss_supcon = supervised_contrastive_loss(
                torch.cat(embeddings, dim=0), torch.cat(labels), supcon_temperature,
            )
            loss = loss_ce + supcon_weight * loss_supcon
            if scaler:
                scaler.scale(loss).backward()
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skip_lr_sched = scale_before > scaler.get_scale()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            history['objective_ce'].append(float(loss_ce.detach()))
            history['objective_supcon'].append(float(loss_supcon.detach()))
            history['objective_total'].append(float(loss.detach()))

        for i, (feats, label, _) in enumerate(train_loader if not supcon_weight else ()):
            is_last_chunk = (len(train_loader) - i) <= remainder
            current_divisor = remainder if (is_last_chunk and remainder > 0) else actual_accum_steps
        
            if scaler:
                with torch.autocast(device_type=device.type):
                    outputs = model(feats.to(device).squeeze(0))
                    logits = outputs[0]
                    loss_ce = criterion(logits, label.to(device))
                    loss = loss_ce / current_divisor

                scaler.scale(loss).backward()
            else:
                outputs = model(feats.to(device).squeeze(0))
                logits = outputs[0]
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
        
        train_loss, train_bacc, _, _, _, _ = evaluate(model, train_eval_loader, device, criterion)
        val_loss, val_bacc, _, _, _, _ = evaluate(model, val_loader, device, criterion)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_bacc'].append(train_bacc)
        history['val_bacc'].append(val_bacc)
        
        pbar.set_postfix({'V_Loss': f"{val_loss:.4f}", 'V_Bacc': f"{val_bacc:.4f}"})
        
        if val_loss < best_val_loss:
            patience_ctr = 0
            best_val_loss = val_loss
            best_val_bacc = val_bacc
            best_weights = snapshot_state(model)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience: 
                stopped_epoch = epoch + 1
                break
            
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    
    _, _, _, _, val_preds, val_probs = evaluate(model, val_loader, device, criterion)
    test_loss, test_bacc, test_auc, test_f1, test_preds, test_probs = evaluate(model, test_loader, device, criterion)
    
    eval_preds = {**val_preds, **test_preds}
    eval_probs = {**val_probs, **test_probs}
    eval_ids = val_ids + test_ids
    
    return best_val_bacc, test_bacc, test_auc, test_f1, stopped_epoch, eval_probs, eval_preds, eval_ids, best_weights, history
    
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

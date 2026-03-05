# train.py
# train.py
import argparse
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn.metrics
from pathlib import Path
from torch.utils.data import DataLoader, Dataset 

from models.models import ZSMIL, TaskRes, CLIPAdapter

# ==========================================
# Dataset Class 
# ==========================================
class WSIDataset(Dataset):
    def __init__(self, features_dir, split="train"):
        self.split_dir = Path(features_dir) / split
        self.pt_files = sorted(list(self.split_dir.glob("*.pt")))
        self.label_map = {"Non-Drinking": 0, "Drinking": 1}

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        data = torch.load(pt_path, map_location="cpu")
        features = data["features"] 
        label = self.label_map[data["binary_category"]]
        return features, label, data["slide_name"]

# ==========================================
# Training Utilities & Main Loop
# ==========================================
def set_random_seeds(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  -> Model Parameters: {trainable:,} trainable / {total:,} total")
    return trainable

def plot_and_save_curves(train_losses, val_losses, train_accs, val_accs, run_name, out_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    
    # --- Loss Subplot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2.5)
    plt.plot(epochs, val_losses, label='Val Loss', linewidth=2.5)
    plt.title('Loss Curve', fontsize=18, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- Accuracy Subplot ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc', linewidth=2.5)
    plt.plot(epochs, val_accs, label='Val BACC', linewidth=2.5)
    plt.title('Accuracy Curve', fontsize=18, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    # Save with 300 DPI for high resolution
    plt.savefig(out_dir / f"{run_name}_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, dataloader, criterion, device):
    """Evaluates the dataset and returns metrics + detailed slide-level predictions."""
    model.eval()
    run_loss = 0.0
    preds, targets = [], []
    slide_details = []
    
    with torch.no_grad():
        for features, label, slide_name in dataloader:
            features, label = features.to(device).squeeze(0), label.to(device)
            logits, _, _ = model(features)
            loss = criterion(logits, label)
            
            prob = logits.softmax(dim=-1)[0, 1].item()
            pred = 1 if prob >= 0.5 else 0
            
            run_loss += loss.item()
            preds.append(pred)
            targets.append(label.item())
            
            str_gt = "Drinking" if label.item() == 1 else "Non-Drinking"
            str_pred = "Drinking" if pred == 1 else "Non-Drinking"
            slide_details.append({
                "slide_name": slide_name[0] if isinstance(slide_name, tuple) else slide_name,
                "ground_truth": str_gt,
                "prediction": str_pred,
                "probability": prob,
                "is_correct": bool(pred == label.item())
            })
            
    avg_loss = run_loss / len(dataloader.dataset)
    bacc = sklearn.metrics.balanced_accuracy_score(targets, preds)
    acc = sklearn.metrics.accuracy_score(targets, preds)
    return avg_loss, bacc, acc, slide_details

def main():
    parser = argparse.ArgumentParser(description="MIL Adapters experimentation")
    parser.add_argument('--adapter', type=str, choices=["ZSMIL", "TaskRes", "CLIPAdapter"], default="CLIPAdapter")
    parser.add_argument('--encoder', type=str, choices=["PLIP", "CONCH"], default="CONCH") 
    parser.add_argument('--aggregator', type=str, choices=["ABMIL", "MeanPool"], default="MeanPool")
    parser.add_argument('--freeze_scale', action='store_true', help="Freeze logit scale to pre-trained value")
    parser.add_argument('--accum_steps', type=int, default=2, help="Number of slides to accumulate gradients over")
    parser.add_argument('--epochs', type=int, default=200) 
    parser.add_argument('--lr', type=float, default=2e-4) 
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--patience', type=int, default=25, help="Early stopping patience")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).resolve().parent
    features_dir = project_root / "outputs" / f"features_{args.encoder}"
    
    # ---------------------------------------------------------
    # DYNAMIC DIRECTORY ORGANIZATION
    # ---------------------------------------------------------
    freeze_str = "_frozen" if args.freeze_scale else ""
    experiment_type = f"{args.adapter}_{args.aggregator}{freeze_str}_{args.encoder}"
    
    results_dir = project_root / "outputs" / "results" / experiment_type
    checkpoints_dir = project_root / "outputs" / "checkpoints" / experiment_type
    
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    text_prototypes = np.load(features_dir / "text_prototypes.npy")

    train_dataset = WSIDataset(features_dir, split="train")
    val_dataset = WSIDataset(features_dir, split="val")
    test_dataset = WSIDataset(features_dir, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    actual_accum_steps = max(1, min(args.accum_steps, len(train_loader)))

    # Trackers for the final aggregation
    all_seeds_test_acc = []
    all_seeds_test_bacc = []
    all_seeds_details = []

    for n_seed in range(args.n_seeds):
        set_random_seeds(n_seed)
        run_name = f"{experiment_type}_seed{n_seed}"
        print(f"\n{'='*60}\n[START] Run: {run_name}\n{'='*60}")

        use_learnable = not args.freeze_scale

        if args.adapter == "ZSMIL":
            model = ZSMIL(text_embeddings=text_prototypes, use_learnable_scale=use_learnable, agg_type=args.aggregator)
        elif args.adapter == "TaskRes":
            model = TaskRes(text_embeddings=text_prototypes, use_learnable_scale=use_learnable, agg_type=args.aggregator)
        elif args.adapter == "CLIPAdapter":
            model = CLIPAdapter(text_embeddings=text_prototypes, use_learnable_scale=use_learnable, agg_type=args.aggregator)
            
        model.to(device)
        trainable_count = count_parameters(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if trainable_count > 0:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3)
            current_epochs = args.epochs
        else:
            optimizer = None
            print("[INFO] Zero trainable parameters detected. Bypassing optimizer and running 1 evaluation epoch.")
            current_epochs = 1 

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        best_val_bacc = -1.0
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = checkpoints_dir / f"{run_name}_best.pth"
        
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        
        for ep in range(current_epochs):
            # --- Training Loop ---
            model.train()
            run_train_acc, run_train_loss = 0.0, 0.0
            if optimizer is not None:
                optimizer.zero_grad()
            
            for i, (features, label, _) in enumerate(train_loader):
                features, label = features.to(device).squeeze(0), label.to(device)
                logits, _, _ = model(features)
                
                loss = criterion(logits, label) / actual_accum_steps
                
                if optimizer is not None:
                    loss.backward()
                    if (i + 1) % actual_accum_steps == 0 or (i + 1) == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                
                pred = logits.softmax(dim=-1).argmax(dim=-1)
                run_train_acc += (pred == label).item()
                run_train_loss += (loss.item() * actual_accum_steps)
                
            epoch_train_acc = run_train_acc / len(train_dataset)
            epoch_train_loss = run_train_loss / len(train_dataset)

            # --- Validation Loop ---
            epoch_val_loss, epoch_val_bacc, _, _ = evaluate_model(model, val_loader, criterion, device)
            
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_accs.append(epoch_train_acc)
            val_accs.append(epoch_val_bacc)
            
            # --- Save Best Model Logic ---
            is_better_acc = epoch_val_bacc > best_val_bacc
            is_equal_acc_better_loss = (epoch_val_bacc == best_val_bacc) and (epoch_val_loss < best_val_loss)
            
            save_msg = ""
            if is_better_acc or is_equal_acc_better_loss:
                if epoch_val_loss < best_val_loss:
                    patience_counter = 0
                best_val_bacc = epoch_val_bacc
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), best_model_path)
                save_msg = f" -> [SAVED] BACC:{best_val_bacc:.4f} L:{best_val_loss:.4f}"
            else:
                patience_counter += 1

            print(f"Ep {ep+1:03d} | TrL: {epoch_train_loss:.4f} | VaL: {epoch_val_loss:.4f} | VaBACC: {epoch_val_bacc:.4f}{save_msg}")

            if optimizer is not None and patience_counter >= args.patience:
                print(f"\n[EARLY STOP] Patience reached at epoch {ep+1}")
                break
                
        plot_and_save_curves(train_losses, val_losses, train_accs, val_accs, run_name, results_dir)

        # --- FINAL TEST EVALUATION (Using Best Checkpoint) ---
        print("\n[INFO] Evaluating Test Set with best checkpoint...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_bacc, test_acc, slide_details = evaluate_model(model, test_loader, criterion, device)
        
        print(f"{'Slide Name':<30} | {'Ground Truth':<15} | {'Prediction':<15} | {'Prob':<5} | {'Match'}")
        print("-" * 80)
        for detail in slide_details:
            match = "✅" if detail['is_correct'] else "❌"
            print(f"{detail['slide_name']:<30} | {detail['ground_truth']:<15} | {detail['prediction']:<15} | {detail['probability']:.2f} | {match}")
        
        print("-" * 80)
        print(f"-> FINAL TEST (Seed {n_seed}) | Acc: {test_acc:.4f} | BACC: {test_bacc:.4f}\n")

        all_seeds_test_acc.append(test_acc)
        all_seeds_test_bacc.append(test_bacc)
        all_seeds_details.append({f"seed_{n_seed}": slide_details})

    # ==========================================
    # FINAL AGGREGATION ACROSS ALL SEEDS
    # ==========================================
    mean_acc = np.mean(all_seeds_test_acc)
    std_acc = np.std(all_seeds_test_acc)
    mean_bacc = np.mean(all_seeds_test_bacc)
    std_bacc = np.std(all_seeds_test_bacc)

    print(f"\n{'='*60}")
    print(f"[AGGREGATE METRICS ACROSS {args.n_seeds} SEEDS]")
    print(f"Experiment: {experiment_type}")
    print(f"Test Accuracy:          {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Test Balanced Accuracy: {mean_bacc:.4f} ± {std_bacc:.4f}")
    print(f"{'='*60}\n")

    # Log everything into a single, comprehensive JSON file
    summary_data = {
        "experiment_config": {
            "adapter": args.adapter,
            "aggregator": args.aggregator,
            "encoder": args.encoder,
            "freeze_scale": args.freeze_scale,
            "learning_rate": args.lr,
            "n_seeds": args.n_seeds
        },
        "aggregate_test_metrics": {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_balanced_accuracy": mean_bacc,
            "std_balanced_accuracy": std_bacc
        },
        "individual_seed_results": all_seeds_details
    }

    summary_path = results_dir / "final_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"[SAVE] Comprehensive run summary saved to {summary_path}")

if __name__ == "__main__":
    main()


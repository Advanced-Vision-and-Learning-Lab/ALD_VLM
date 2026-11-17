"""Receiver operating characteristic curve plotting."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def save_roc_curve(targets, probs, save_path, title_prefix=""):
    """
    Generate and save a ROC curve.
    """
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6), dpi=300)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7, label='Random Chance Baseline') 
    
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'{title_prefix} ROC Curve', fontsize=16, pad=15)
    
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

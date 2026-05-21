"""
generate_latex_table.py
"""

import os
import re
from pathlib import Path

EXPERIMENTS = [
    ("ZSMIL", "zsmil_d64"),
    ("TaskRes ($\\alpha=0.25$)", "taskres_a0.25_d64")
]

POOLS = [("Mean", "mean"), ("LSE", "lse")]
SHOTS = [1, 2, 4]
ITERATIONS = 10
STANDALONE_DIR = Path("outputs")

def parse_metrics(filepath):
    if not filepath.exists():
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()
        
    if "OVERALL METRICS" in content:
        content = content.split("OVERALL METRICS")[1]
    else:
        return None 
        
    metrics = {}
    param_match = re.search(r"Trainable Parameters:\s+(\d+)", content)
    if param_match:
        metrics['params'] = int(param_match.group(1)) / 1000.0
        
    bacc_match = re.search(r"Test BACC:\s+([0-9.]+)\s+±\s+([0-9.]+)", content)
    if bacc_match:
        metrics['bacc'] = (float(bacc_match.group(1))*100, float(bacc_match.group(2))*100)
        
    auc_match = re.search(r"Test AUC-ROC:\s+([0-9.]+)\s+±\s+([0-9.]+)", content)
    if auc_match:
        metrics['auc'] = (float(auc_match.group(1))*100, float(auc_match.group(2))*100)
        
    f1_match = re.search(r"Test F1-Score:\s+([0-9.]+)\s+±\s+([0-9.]+)", content)
    if f1_match:
        metrics['f1'] = (float(f1_match.group(1))*100, float(f1_match.group(2))*100)

    return metrics if len(metrics) >= 3 else None

def build_latex_table():
    results = {}
    rows = []
    
    row_idx = 0
    for display_name, base_prefix in EXPERIMENTS:
        for pool_display, pool_suffix in POOLS:
            rows.append(f"{display_name} ({pool_display})")
            
            for shot in SHOTS:
                folder_name = f"{base_prefix}_{pool_suffix}_{shot}shot_{ITERATIONS}iter"
                filepath = STANDALONE_DIR / folder_name / "summary_metrics.txt"
                metrics = parse_metrics(filepath)

                if shot == 1:
                    if "params" not in results:
                        results["params"] = {}
                    results["params"][row_idx] = metrics.get('params') if metrics else None
                
                for metric_name in ['bacc', 'auc', 'f1']:
                    col_key = f"{shot}shot_{metric_name}"
                    if col_key not in results:
                        results[col_key] = {}
                    results[col_key][row_idx] = metrics[metric_name] if metrics else None
                        
            row_idx += 1

    best_indices = {}
    for col_key in results:
        if 'params' in col_key:
            continue
        valid_means = {r_idx: vals[0] for r_idx, vals in results[col_key].items() if vals is not None}
        if len(valid_means) == 0:
            best_indices[col_key] = (None, None)
            continue

        sorted_items = sorted(valid_means.items(), key=lambda x: x[1], reverse=True)
        top1_idx = sorted_items[0][0]
        top2_idx = sorted_items[1][0] if len(sorted_items) > 1 else None
        best_indices[col_key] = (top1_idx, top2_idx)

    latex = [
        r"\begin{table}[hbt!]",
        r"\centering",
        r"\caption{1, 2, and 4-Shot WSI Classification Performance comparing Mean vs. LSE Pooling.}",
        r"\label{tab:lse_ablation}",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{l c ccc ccc ccc}",
        r"\toprule",
        r"\multirow{2}{*}{Method} & \multirow{2}{*}{\# Params (K)} & \multicolumn{3}{c}{\textbf{1-Shot}} & \multicolumn{3}{c}{\textbf{2-Shot}} & \multicolumn{3}{c}{\textbf{4-Shot}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}",
        r" & & BACC & AUC & F1 & BACC & AUC & F1 & BACC & AUC & F1 \\",
        r"\midrule"
    ]

    for i, row_name in enumerate(rows):
        param_val = results.get("params", {}).get(i)
        param_str = f"{param_val:.1f}" if param_val is not None else "-"
        row_str = f"{row_name} & {param_str}"
        
        for shot in SHOTS:
            for metric in ['bacc', 'auc', 'f1']:
                col_key = f"{shot}shot_{metric}"
                val_tuple = results[col_key].get(i)
                
                if val_tuple is None:
                    row_str += " & - "
                else:
                    mean, std = val_tuple
                    top1, top2 = best_indices.get(col_key, (None, None))
                    
                    if i == top1:
                        cell_str = f"$\\mathbf{{{mean:.1f} \\pm {std:.1f}}}$"
                    elif i == top2:
                        cell_str = f"$\\underline{{{mean:.1f} \\pm {std:.1f}}}$"
                    else:
                        cell_str = f"${mean:.1f} \\pm {std:.1f}$"
                        
                    row_str += f" & {cell_str}"
        row_str += " \\\\"
        latex.append(row_str)

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}"
    ])

    return "\n".join(latex)

if __name__ == "__main__":
    print(build_latex_table())

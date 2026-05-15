"""
generate_latex_table.py
"""

import os
import re
from pathlib import Path

# Configuration for Table 1 (All Methods)
EXPERIMENTS_ALL = [
    ("ABMIL", "zsmil_d64"),
    ("ABMIL + Adapter", "abmil_adapter_r4_d64"),
    ("ABMIL + TaskRes", "taskres_a0.25_d64"),
    ("ABMIL + NL-TaskRes", "nl_taskres_r8_a0.25_d64"), 
    ("ABMIL + Affine TaskRes", "affine_taskres_a0.25_d64"), 
    ("ABMIL + Dual Adapt", "dual_adapt_r4_a0.25_d64")
]

# Configuration for Table 2 (Excluding Dual Adapt)
EXPERIMENTS_NO_DUAL = [
    ("ABMIL", "zsmil_d64"),
    ("ABMIL + Adapter", "abmil_adapter_r4_d64"),
    ("ABMIL + TaskRes", "taskres_a0.25_d64"),
    ("ABMIL + NL-TaskRes", "nl_taskres_r8_a0.25_d64"),  
    ("ABMIL + Affine TaskRes", "affine_taskres_a0.25_d64") # <-- ADDED THIS LINE HERE
]

POOLS = [("Mean", "mean"), ("LSE", "lse")]
SHOTS = [1, 4]
ITERATIONS = 10
STANDALONE_DIR = Path("outputs")

def parse_metrics(filepath):
    """Reads the summary file and extracts overall params, BACC, AUC, and F1 means/stds."""
    if not filepath.exists():
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()
        
    # CRITICAL FIX: Isolate the final metrics block to avoid reading iteration 1
    if "OVERALL METRICS" in content:
        content = content.split("OVERALL METRICS")[1]
    else:
        return None # Failsafe if the file didn't finish writing
        
    metrics = {}
    
    param_match = re.search(r"Trainable Parameters:\s+(\d+)", content)
    if param_match:
        metrics['params'] = int(param_match.group(1)) / 1000.0
        
    # Extract values and multiply by 100 for percentage format
    bacc_match = re.search(r"Test BACC:\s+([0-9.]+)\s+±\s+([0-9.]+)", content)
    if bacc_match:
        metrics['bacc'] = (float(bacc_match.group(1))*100, float(bacc_match.group(2))*100)
        
    auc_match = re.search(r"Test AUC-ROC:\s+([0-9.]+)\s+±\s+([0-9.]+)", content)
    if auc_match:
        metrics['auc'] = (float(auc_match.group(1))*100, float(auc_match.group(2))*100)
        
    f1_match = re.search(r"Test F1-Score:\s+([0-9.]+)\s+±\s+([0-9.]+)", content)
    if f1_match:
        metrics['f1'] = (float(f1_match.group(1))*100, float(f1_match.group(2))*100)
        
    # CRITICAL FIX: Use >= 3 to allow passing if params is present (length = 4)
    return metrics if len(metrics) >= 3 else None

def build_latex_table(experiment_list, table_label, table_caption):
    """Generates a fully formatted LaTeX table for a given list of experiments."""
    results = {}
    rows = []
    
    row_idx = 0
    # 1. Gather all metrics from directories
    for display_name, base_prefix in experiment_list:
        for pool_display, pool_suffix in POOLS:
            rows.append(f"{display_name} ({pool_display})")
            
            for shot in SHOTS:
                folder_name = f"{base_prefix}_{pool_suffix}_{shot}shot_{ITERATIONS}iter"
                filepath = STANDALONE_DIR / folder_name / "summary_metrics.txt"
                
                metrics = parse_metrics(filepath)
                
                # CRITICAL FIX: Explicitly store params for the 1-shot column
                if shot == 1:
                    if "1shot_params" not in results:
                        results["1shot_params"] = {}
                    results["1shot_params"][row_idx] = metrics.get('params') if metrics else None
                
                for metric_name in ['bacc', 'auc', 'f1']:
                    col_key = f"{shot}shot_{metric_name}"
                    if col_key not in results:
                        results[col_key] = {}
                    
                    results[col_key][row_idx] = metrics[metric_name] if metrics else None
                        
            row_idx += 1

    # 2. Find Top 1 and Top 2 indices for bolding and underlining
    best_indices = {}
    for col_key in results:
        # Ignore the params column for top-1/top-2 styling logic
        if 'params' in col_key:
            continue
            
        valid_means = {r_idx: vals[0] for r_idx, vals in results[col_key].items() if vals is not None}
        
        if len(valid_means) == 0:
            best_indices[col_key] = (None, None)
            continue
            
        # Sort descending to find max values
        sorted_items = sorted(valid_means.items(), key=lambda x: x[1], reverse=True)
        top1_idx = sorted_items[0][0]
        top2_idx = sorted_items[1][0] if len(sorted_items) > 1 else None
        
        best_indices[col_key] = (top1_idx, top2_idx)

    # 3. Construct LaTeX block
    latex = [
        r"\begin{table}[hbt!]",
        r"\centering",
        f"\\caption{{{table_caption}}}",
        f"\\label{{{table_label}}}",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{l c ccc ccc}",
        r"\toprule",
        r"\multirow{2}{*}{Method} & \multirow{2}{*}{\# Params (K)} & \multicolumn{3}{c}{\textbf{1-Shot}} & \multicolumn{3}{c}{\textbf{4-Shot}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r" & & BACC & AUC & F1 score & BACC & AUC & F1 score \\",
        r"\midrule"
    ]

    for i, row_name in enumerate(rows):
        # Extract the parameter count (using 1-shot as the baseline since params don't change per shot)
        param_val = results.get("1shot_params", {}).get(i)
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

def generate_all_tables():
    # Generate Table 1
    table_all = build_latex_table(
        experiment_list=EXPERIMENTS_ALL,
        table_label="tab:few_shot_results_all",
        table_caption="1-Shot and 4-Shot WSI Classification Performance (All Methods). Bold values indicate the highest performance, while underlined values denote the second highest."
    )
    
    # Generate Table 2
    table_no_dual = build_latex_table(
        experiment_list=EXPERIMENTS_NO_DUAL,
        table_label="tab:few_shot_results_no_dual",
        table_caption="1-Shot and 4-Shot WSI Classification Performance (Excluding Dual Adapt). Bold values indicate the highest performance, while underlined values denote the second highest."
    )
    
    print("%%% TABLE 1: ALL METHODS %%%")
    print(table_all)
    print("\n\n")
    print("%%% TABLE 2: NO DUAL ADAPT %%%")
    print(table_no_dual)

if __name__ == "__main__":
    generate_all_tables()

#!/bin/bash

# Define the exact 9 combinations to match the new Table: "model max_pool_head learning_rate"
runs=(
    # --- BASELINES ---
    "max_pool none 1e-3"
    "max_pool adapter 4e-4"
    "max_pool taskres 1e-3"
    
    # --- ORIGINAL VISUAL ABMIL ---
    "zsmil none 1e-3"
    "taskres none 1e-3"
    "abmil_adapter none 4e-4"
    
    # --- NOVEL CROSS-ATTENTION ---
    "cross_attn none 1e-3"
    "cross_attn_adapter none 4e-4"
    "cross_attn_taskres none 1e-3"
)

for shots in 1 4 8; do
    echo "======================================================="
    echo "  EVALUATING: ${shots} SHOT(S) | 10 ITERS | 3 SEEDS"
    echo "======================================================="

    for run in "${runs[@]}"; do
        # Parse the string into variables
        read -r model head lr <<< "$run"
        
        echo "--> Running $model with head=$head (LR: $lr)"
        python demo.py \
            --model "$model" \
            --max_pool_head "$head" \
            --epochs 250 \
            --lr "$lr" \
            --reduction 8 \
            --abmil_d 64 \
            --ortho_weight 0.1 \
            --shots "$shots" \
            --iterations 10 \
            --dataset_pct 1.0 \
            --seeds 3 \
            --patience 50
    done
done



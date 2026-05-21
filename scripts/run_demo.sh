#!/bin/bash

SHOTS=(1 2 4)
POOLS=("mean" "lse")
MODELS=("zsmil" "taskres")

ALPHA=0.25
MAX_POOL_HEAD="none"
EPOCHS=250
LR=5e-4
REDUCTION=4
ABMIL_D=64
ITERATIONS=10
DATASET_PCT=1.0
SEEDS=3
PATIENCE=50
TASKRES_DROP=0.1  

echo "Starting Few-Shot WSI Experiments (Exhaustive Arguments)..."

for SHOT in "${SHOTS[@]}"; do
    for POOL in "${POOLS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            echo "======================================================================"
            echo "Running: Model=$MODEL | Shots=$SHOT | Pool=$POOL"
            echo "======================================================================"

            python demo.py \
                --model "$MODEL" \
                --alpha "$ALPHA" \
                --max_pool_head "$MAX_POOL_HEAD" \
                --text_pool "$POOL" \
                --epochs "$EPOCHS" \
                --lr "$LR" \
                --reduction "$REDUCTION" \
                --abmil_d "$ABMIL_D" \
                --shots "$SHOT" \
                --iterations "$ITERATIONS" \
                --dataset_pct "$DATASET_PCT" \
                --seeds "$SEEDS" \
                --patience "$PATIENCE" \
                --taskres_drop "$TASKRES_DROP"
                
        done
    done
done

echo "All specified experiments completed!"

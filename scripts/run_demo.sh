#!/bin/bash

# Configuration Arrays
SHOTS=(1 4)
POOLS=("mean" "lse")
MODELS=("zsmil" "abmil_adapter" "taskres" "dual_adapt")

# Fixed Hyperparameters
ABMIL_D=64
REDUCTION=4
LR=5e-4
ITERATIONS=10
SEEDS=3
PATIENCE=50
EPOCHS=250

echo "Starting Few-Shot WSI Experiments..."

for SHOT in "${SHOTS[@]}"; do
    for POOL in "${POOLS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            echo "======================================================================"
            echo "Running: Model=$MODEL | Shots=$SHOT | Text Pool=$POOL"
            echo "======================================================================"

            python demo.py \
                --model "$MODEL" \
                --text_pool "$POOL" \
                --shots "$SHOT" \
                --abmil_d "$ABMIL_D" \
                --reduction "$REDUCTION" \
                --lr "$LR" \
                --iterations "$ITERATIONS" \
                --seeds "$SEEDS" \
                --patience "$PATIENCE" \
                --epochs "$EPOCHS"
                
        done
    done
done

echo "All experiments completed!"

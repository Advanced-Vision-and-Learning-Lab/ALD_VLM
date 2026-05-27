#!/bin/bash

SHOTS=(1 2 4)
POOLS=("mean" "lse")
MODELS=("max_pool" "zsmil" "abmil_adapter" "abmil_adapter_dim8" "taskres")

ALPHA=0.25
MAX_POOL_HEAD="none"
EPOCHS=250
LR=1e-3
REDUCTION=4
ABMIL_D=64
ITERATIONS=10
DATASET_PCT=1.0
SEEDS=3
PATIENCE=50
TASKRES_DROP=0.0  

echo "Starting Few-Shot WSI Experiments..."

for SHOT in "${SHOTS[@]}"; do
    for POOL in "${POOLS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            
            RUN_MODEL="$MODEL"
            RUN_REDUCTION="$REDUCTION"
            
            if [[ "$MODEL" == "abmil_adapter_dim8" ]]; then
                RUN_MODEL="abmil_adapter"
                RUN_REDUCTION=64
            fi

            echo "======================================================================"
            echo "Running: Model=$MODEL | Shots=$SHOT | Pool=$POOL | Reduction=$RUN_REDUCTION"
            echo "======================================================================"

            python demo.py \
                --model "$RUN_MODEL" \
                --alpha "$ALPHA" \
                --max_pool_head "$MAX_POOL_HEAD" \
                --text_pool "$POOL" \
                --epochs "$EPOCHS" \
                --lr "$LR" \
                --reduction "$RUN_REDUCTION" \
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

echo "All experiments completed!"

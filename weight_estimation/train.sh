#!/bin/bash
#SBATCH -A USERNAME
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3-0:00:00
#SBATCH --output=NET7_BASELINE.txt
#SBATCH --partition=u22
#SBATCH -c 8
#SBATCH --job-name=net7_baseline
#SBATCH --nodelist=gnode080

set -e


export WANDB_CACHE_DIR=/path/to/wandb_cache
export WANDB_DIR=/path/to/wandb_dir
export WANDB_ARTIFACT_DIR=/path/to/wandb_artifacts

# === Baseline (RGB only, no geom) ===
ROOT_DIR=/path/to/train_dataset
DEPTH_DIR=/path/to/train_depth_maps
VAL_DIR=/path/to/val_dataset
VAL_DEPTH_DIR=/path/to/val_depth_maps
WEIGHT_JSON=all_food_data.json
INDEX_MAP=index_map.json
BATCH_SIZE=4
EPOCHS=100
LR=1e-3
BACKBONE="resnet50"
ROI_RES=7
MODALITY="rgb"
ATTENTION="none"
GEOM="none"
OUT_DIR="./output/baseline"

# mkdir -p "${OUT_DIR}"

echo "Running Baseline (RGB only, no geometric features) unFROZEN"
python train.py \
    --root_dir "${ROOT_DIR}" \
    --depth_dir "${DEPTH_DIR}" \
    --val_dir "${VAL_DIR}" \
    --val_depth_dir "${VAL_DEPTH_DIR}" \
    --weight_json "${WEIGHT_JSON}" \
    --index_map "${INDEX_MAP}" \
    --out_dir "${OUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --backbone "${BACKBONE}" \
    --roi_res "${ROI_RES}" \
    --modality "${MODALITY}" \
    --attention "${ATTENTION}" \
    --geom "${GEOM}" \
    --unfreeze 

echo "✅ Baseline run complete: results in ${OUT_DIR}"

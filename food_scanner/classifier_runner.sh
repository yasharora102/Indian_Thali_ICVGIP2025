
# ------------------------------------------------------------------------------
# Classifier Runner Script
# 
# Please update the paths below to match your local setup.
# ------------------------------------------------------------------------------

CONFIG="config.yaml"
INPUT_DIR="/path/to/weight_data/patch_embeddings"
PROTOTYPE_DIR="/path/to/prototypes"
ORIG_IMAGES_DIR="/path/to/weight_data/images"
OUTPUT_BASE="/path/to/results"
GT_DIR="/path/to/weight_data/masks"
MENU_JSON="config/menu.json"
WORKERS=12

# ------------------------------------------------------------------------------
# 1. Global Classification (Menu-Filtered)
# ------------------------------------------------------------------------------

# k=1
python src/classify_patches_global.py \
    --config "$CONFIG" \
    --input_dir "$INPUT_DIR" \
    --orig_images_dir "$ORIG_IMAGES_DIR" \
    --workers $WORKERS --knn_k 1 \
    --prototype_dir "$PROTOTYPE_DIR" \
    --menu_json "$MENU_JSON" \
    --output_dir "$OUTPUT_BASE/plates/global/knn_1/"

python src/eval_blob_indexbased.py \
    --parent_dir "$OUTPUT_BASE/plates/global/knn_1/"  \
    --gt_dir "$GT_DIR" \
    --output_dir "$OUTPUT_BASE/plates/global/knn_1_majority_results/" \
    --min_area 2200 --iou_threshold 0.5 --cmap OrRd --workers $WORKERS \
    --pooling majority

python src/eval_blob_indexbased.py \
    --parent_dir "$OUTPUT_BASE/plates/global/knn_1/"  \
    --gt_dir "$GT_DIR" \
    --output_dir "$OUTPUT_BASE/plates/global/knn_1_pooled_results/" \
    --min_area 2200 --iou_threshold 0.5 --cmap OrRd --workers $WORKERS \
    --pooling pooled

# k=3
python src/classify_patches_global.py \
    --config "$CONFIG" \
    --input_dir "$INPUT_DIR" \
    --orig_images_dir "$ORIG_IMAGES_DIR" \
    --workers $WORKERS --knn_k 3 \
    --prototype_dir "$PROTOTYPE_DIR" \
    --menu_json "$MENU_JSON" \
    --output_dir "$OUTPUT_BASE/plates/global/knn_3/"

python src/eval_blob_indexbased.py \
    --parent_dir "$OUTPUT_BASE/plates/global/knn_3/"  \
    --gt_dir "$GT_DIR" \
    --output_dir "$OUTPUT_BASE/plates/global/knn_3_majority_results/" \
    --min_area 2200 --iou_threshold 0.5 --cmap OrRd --workers $WORKERS \
    --pooling majority

python src/eval_blob_indexbased.py \
    --parent_dir "$OUTPUT_BASE/plates/global/knn_3/"  \
    --gt_dir "$GT_DIR" \
    --output_dir "$OUTPUT_BASE/plates/global/knn_3_pooled_results/" \
    --min_area 2200 --iou_threshold 0.5 --cmap OrRd --workers $WORKERS \
    --pooling pooled

# (Repeat for k=5 if needed)
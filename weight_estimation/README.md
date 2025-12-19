## Data Download
Please find the Weight Estimation Dataset below and download it.
```sh
wget http://cvit.iiit.ac.in/images/datasets/food_nutrition/WED.tar.gz
```

Please find the baseline checkpoint in the relases and download it from there.
Baseline Model:  [LINK](https://github.com/yasharora102/Indian_Thali/releases/download/models/checkpoint_baseline.pth).

## Index Map
The index map for WED dataset can be found in this directory named [`index_map.json`](https://github.com/yasharora102/Indian_Thali/blob/main/02_weight/index_map.json)

## Weight Data
The individual weights for each ROI can be found under [`all_food_data.json`](https://github.com/yasharora102/Indian_Thali/blob/main/02_weight/all_food_data.json)

## Training 
To train the baseline, set the correct paths in network7_train.sh and run it. You may need to set up a wandb account for it.

```sh
WANDB_CACHE_DIR=""
WANDB_DIR=""
WANDB_ARTIFACT_DIR=""

# === Baseline (RGB only, no geom) ===
ROOT_DIR=/scratch/weight-est/Weight-estimation-split_realigned_resized/train/ 
DEPTH_DIR=/scratch/weight-est/train_depthmaps/
VAL_DIR=/scratch/weight-est/Weight-estimation-split_realigned_resized/test
VAL_DEPTH_DIR=/scratch/weight-est/test_depthmaps/
WEIGHT_JSON=all_food_data.json
INDEX_MAP=index_map.json
```

## Evaluation

To evaluate the baseline:
Maintain the flags as in the training setup inside evaluate.sh, and run it with the checkpoint path.

```sh
    --root_dir "/scratch/weight-est/Weight-estimation-split_realigned/test" \
    --depth_dir /scratch/weight-est/test_depthmaps/ \
    --weight_json all_food_data.json \
    --index_map index_map.json \
    --checkpoint /scratch/weight-est/Net7/unfrozen/baseline_CBAM/best_model.pth  \
    --out_csv_class FINAL_OUTS/RERUN_FINAL/BASELINE_depthmod/out_csv_class.csv \
    --out_csv_sample FINAL_OUTS/RERUN_FINAL/BASELINE_depthmod/out_csv_sample.csv \
    --modality rgb \
    --attention none \
    --geom_type none \
    --backbone resnet50 \
```


Run it like this
```sh
bash eval_net7.sh
```

## Download data
### ITD
```bash
wget https://cvit.iiit.ac.in/images/datasets/food_nutrition/ITD.tar.gz
```
---
Ensure you have MMSegmentation and MMDetection installed.

You need to change the paths in the `configs/_base_/datasets/your_dataset_big.py` file to point to the correct locations of your datasets.

You also need to ensure that the number of classes in the model config file matches the number of classes in your dataset.

For training, you can use the following command:
```bash
python tools/train.py configs/your_config_file.py
```
To test the model, use:
```bash
# mask2former_R50 EXAMPLE
python tools/test.py \
    configs/mask2former/mask2former_r50_8xb2-80k_MYDATA-512x1024.py \
    /scratch/seg_benchmark/NEW/mask2former_R50_320K/iter_160000.pth \
    --out /scratch/seg_benchmark/NEW/mask2former_R50_320K/results_160K \
    --work-dir /scratch/seg_benchmark/mmseg_work-dir
```

You can test the various segmentation accuracies useing the script in the `tools/analysis_tools/` directory.
`segmentation/examples/benchmark_mask2former.sh` and `segmentation/tools/analysis_tools/run_metrics.sh` have some example commands.

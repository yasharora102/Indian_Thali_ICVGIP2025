
# FCN
python tools/test.py \
    configs/fcn/fcn_r50-d8_4xb2-80k_MYDATA-512x1024.py \
    /scratch/seg_benchmark/NEW/FCN_R50-resized_320K/iter_160000.pth \
    --out /scratch/seg_benchmark/NEW/FCN_R50-resized_320K/results_160K \
    --work-dir /scratch/seg_benchmark/mmseg_work-dir



# SegFormer
python tools/test.py \
    configs/segformer/segformer_mit-b3_8xb1-160k_mydataset-512x1024.py \
    /scratch/seg_benchmark/NEW/segformer-mitb3-FUll-resized_seed_320K/iter_160000.pth \
    --out /scratch/seg_benchmark/NEW/segformer-mitb3-FUll-resized_seed_320K/results_160K \
    --work-dir /scratch/seg_benchmark/mmseg_work-dir




#deeplab
python tools/test.py \
    configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024_modified.py \
    /scratch/seg_benchmark/NEW/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/iter_160000.pth \
    --out /scratch/seg_benchmark/NEW/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/results_160K \
    --work-dir /scratch/seg_benchmark/mmseg_work-dir
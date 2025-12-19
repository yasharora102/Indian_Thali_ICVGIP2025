# # mask2former_swinT
python tools/test.py \
    configs/mask2former/mask2former_swin-t_8xb2-80k_MYDATA-512x1024.py \
    /scratch/seg_benchmark/NEW/mask2former_swin_T_seed_320K/iter_160000.pth \
    --out /scratch/seg_benchmark/NEW/mask2former_swin_T_seed_320K/results_160K \
    --work-dir /scratch/seg_benchmark/mmseg_work-dir


# mask2former_R50
python tools/test.py \
    configs/mask2former/mask2former_r50_8xb2-80k_MYDATA-512x1024.py \
    /scratch/seg_benchmark/NEW/mask2former_R50_320K/iter_160000.pth \
    --out /scratch/seg_benchmark/NEW/mask2former_R50_320K/results_160K \
    --work-dir /scratch/seg_benchmark/mmseg_work-dir

# segnext
python tools/test.py \
    configs/segnext/segnext_mscan-l_1xb16-adamw-80k_mYDATA-512x512.py \
    /scratch/seg_benchmark/NEW/segnext_l-resized_seed_320K/iter_160000.pth \
    --out /scratch/seg_benchmark/NEW/segnext_l-resized_seed_320K/results_160K \
    --work-dir /scratch/seg_benchmark/mmseg_work-dir


# # FCN
# python tools/test.py \
#     configs/fcn/fcn_r50-d8_4xb2-80k_MYDATA-512x1024.py \
#     /scratch/seg_benchmark/NEW/FCN_R50-resized_320K/FCN_R50-resized_320K/iter_160000.pth \
#     --out /scratch/seg_benchmark/NEW/FCN_R50-resized_320K/FCN_R50-resized_320K/results_160K \
#     --work-dir /scratch/seg_benchmark/mmseg_work-dir



# # SegFormer
# python tools/test.py \
#     configs/segformer/segformer_mit-b3_8xb1-160k_mydataset-512x1024.py \
#     /scratch/seg_benchmark/NEW/segformer-mitb3-FUll-resized_seed_320K/segformer-mitb3-FUll-resized_seed_320K/iter_160000.pth \
#     --out /scratch/seg_benchmark/NEW/segformer-mitb3-FUll-resized_seed_320K/segformer-mitb3-FUll-resized_seed_320K/results_160K \
#     --work-dir /scratch/seg_benchmark/mmseg_work-dir




# #deeplab
# python tools/test.py \
#     configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024_modified.py \
#     /scratch/seg_benchmark/NEW/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/iter_160000.pth \
#     --out /scratch/seg_benchmark/NEW/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/results_160K \
#     --work-dir /scratch/seg_benchmark/mmseg_work-dir
# # mask2former-swinT
# python tools/test.py \
#     configs/mask2former/mask2former_swin-t_8xb2-80k_MYDATA-512x1024.py \
#     /scratch/seg_benchmark/FINAL_seg_full_redone_redo/mask2former_swin_T_seed/iter_80000.pth  \
#     --out /scratch/seg_benchmark/FINAL_seg_full_redone_redo/mask2former_swin_T_seed/results \
#     --work-dir /scratch/seg_benchmark/mmseg_work-dir



# zip -r Mask2Former_SwinT.zip /scratch/seg_benchmark/FINAL_seg_full_redone_redo/mask2former_swin_T_seed/results
# zip -r SegNeXt.zip /scratch/seg_benchmark/FINAL_seg_full_redone/segnext_l-resized_seed/results
# zip -r SegFormer.zip /scratch/seg_benchmark/FINAL_seg_full_redone/segformer-mitb3-FUll-resized_seed/segformer-mitb3-FUll-resized_seed/results
# zip -r DeepLab.zip /scratch/seg_benchmark/FINAL_seg_full_redone/mmseg_deeplabv3p_80k_big_RESIZED_fullv1_seed/results
# zip -r FCN.zip /scratch/seg_benchmark/FINAL_seg_full_redone-seed/FCN_R50-resized/results





# mask2former-r50
# python tools/test.py \
#     configs/mask2former/mask2former_r50_8xb2-80k_MYDATA-512x1024.py \
#     /scratch/seg_benchmark/FINAL_seg_full_redone_redo_SEED/mask2former_R50/iter_80000.pth  \
#     --out /scratch/seg_benchmark/FINAL_seg_full_redone_redo_SEED/mask2former_R50/results \
#     --work-dir /scratch/seg_benchmark/mmseg_work-dir


# #segformer-mitb3
# python tools/test.py \
#     configs/segformer/segformer_mit-b3_8xb1-160k_mydataset-512x1024.py \
#     /scratch/seg_benchmark/FINAL_seg_full_redone/segformer-mitb3-FUll-resized_seed/segformer-mitb3-FUll-resized_seed/iter_80000.pth  \
#     --out /scratch/seg_benchmark/FINAL_seg_full_redone/segformer-mitb3-FUll-resized_seed/segformer-mitb3-FUll-resized_seed/results \
#     --work-dir /scratch/seg_benchmark/mmseg_work-dir




# # segnext-L
# python tools/test.py \
#     configs/segnext/segnext_mscan-l_1xb16-adamw-80k_mYDATA-512x512.py \
#     /scratch/seg_benchmark/FINAL_seg_full_redone/segnext_l-resized_seed/iter_80000.pth  \
#     --out /scratch/seg_benchmark/FINAL_seg_full_redone/segnext_l-resized_seed/results \
#     --work-dir /scratch/seg_benchmark/mmseg_work-dir



 
# # fcn50
#     python tools/test.py \
#     configs/fcn/fcn_r50-d8_4xb2-80k_MYDATA-512x1024.py \
#     /scratch/seg_benchmark/FINAL_seg_full_redone-seed/FCN_R50-resized/iter_80000.pth  \
#     --out /scratch/seg_benchmark/FINAL_seg_full_redone-seed/FCN_R50-resized/results \
#     --work-dir /scratch/seg_benchmark/mmseg_work-dir

# # deeplab
#     python tools/test.py \
#     configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024_modified.py \
#     /scratch/seg_benchmark/FINAL_seg_full_redone/mmseg_deeplabv3p_80k_big_RESIZED_fullv1_seed/iter_80000.pth  \
#     --out /scratch/seg_benchmark/FINAL_seg_full_redone/mmseg_deeplabv3p_80k_big_RESIZED_fullv1_seed/results \
#     --work-dir /scratch/seg_benchmark/mmseg_work-dir


    
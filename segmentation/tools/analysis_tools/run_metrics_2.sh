# # python eval_blob.py --gt-dir /scratch/seg_benchmark/splits_flat/test/masks/ \
# #     --pred-dir /scratch/seg_benchmark/NEW/mask2former_swin_T_seed_320K/mask2former_swin_T_seed_320K/results_40K/ \
# #     --output_dir ../../result/blob_metrics/deeplab


# python pix-acc.py --gt-dir /scratch/seg_benchmark/splits_flat/test/masks/ \
#     --class-map /home2/yasharora120/segmentation_benchmark/ClipSegV3/new_label_map.json --workers 8 \
#     --pred-dir /scratch/seg_benchmark/NEW/segnext_l-resized_seed_320K/segnext_l-resized_seed_320K/results_40K/ 


# # python pix-acc_agnostic.py --gt-dir /scratch/seg_benchmark/splits_flat/test/masks/ \
# #     --class-map /home2/yasharora120/segmentation_benchmark/ClipSegV3/new_label_map.json --workers 8 \
# #     --pred-dir /scratch/seg_benchmark/NEW/mask2former_swin_T_seed_320K/mask2former_swin_T_seed_320K/results_40K/ 





python eval_blob.py --gt-dir /scratch/seg_benchmark/splits_flat/test/masks/ \
    --pred-dir /scratch/seg_benchmark/NEW/FCN_R50-resized_320K/results_160K/ \
    --output_dir ../../result_NEW/blob_metrics/fcn


python eval_blob.py --gt-dir /scratch/seg_benchmark/splits_flat/test/masks/ \
    --pred-dir /scratch/seg_benchmark/NEW/deeplabv3p_80k_big_RESIZED_fullv1_seed_320K/results_160K/ \
    --output_dir ../../result_NEW/blob_metrics/deeplab


python eval_blob.py --gt-dir /scratch/seg_benchmark/splits_flat/test/masks/ \
    --pred-dir /scratch/seg_benchmark/NEW/segformer-mitb3-FUll-resized_seed_320K/results_160K/ \
    --output_dir ../../result_NEW/blob_metrics/segt
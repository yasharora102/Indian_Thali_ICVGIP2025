python evaluate.py \
    --root_dir "/path/to/test_dataset" \
    --depth_dir /path/to/test_depth_maps/ \
    --weight_json all_food_data.json \
    --index_map index_map.json \
    --checkpoint /path/to/best_model.pth  \
    --out_csv_class FINAL_OUTS/out_csv_class.csv \
    --out_csv_sample FINAL_OUTS/out_csv_sample.csv \
    --modality rgb \
    --attention cbam \
    --geom_type none \
    --backbone resnet50 \
    --roi_res 7 
    

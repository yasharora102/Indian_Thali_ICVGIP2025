# mmseg/configs/food/food_segformer_1024x1024.py
from mmseg.datasets.food_dataset import FoodDataset

_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# 1) Dataset settings
dataset_type = 'FoodDataset'
data_root = 'data/food/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

crop_size = (1024, 1024)

# pipelines
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),  # looks for gt_semantic_seg in masks_idx/
    dict(type='Resize', img_scale=(1024,1024), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1024,1024), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='masks_idx/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='masks_idx/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='masks_idx/val',
        pipeline=test_pipeline),
)

# 2) Model: adjust number of classes
model = dict(
    decode_head=dict(num_classes=len(FoodDataset.CLASSES)),
    auxiliary_head=dict(num_classes=len(FoodDataset.CLASSES))
)

# 3) Learning policy overrides (if needed)
optimizer = dict(lr=6e-05)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)

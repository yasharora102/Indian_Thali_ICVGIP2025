# _base_ = [
#     '../_base_/models/segformer_mit-b0.py',
#     '../_base_/default_runtime.py',
#     '../_base_/schedules/schedule_160k.py',
#       '../_base_/datasets/ade20k.py',        # brings in img_norm_cfg

# ]

# # 1. shared image size
# crop_size = (512, 512)
# data_preprocessor = dict(size=crop_size)

# # 2. point to your data root and class palette
# dataset_type = 'CustomDataset'
# data_root = '/home2/yasharora120/segmentation_benchmark/data/food'  # e.g. '/home/user/food_seg'

# # number of classes = number of rows in your colormap.csv
# num_classes =  61 # replace palette with your actual list

# # 3. load / transform pipelines (you can copy pipelines from ADE20K base)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=crop_size, keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='Normalize', **_base_.img_norm_cfg),  # from model base
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img','gt_semantic_seg']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=crop_size,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **_base_.img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ]
#     )
# ]

# # 4. dataset overrides
# data = dict(
#     samples_per_gpu=2,  # adjust batch size
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='train/images',
#         ann_dir='train/ann_masks',
#         split='train.txt',
#         pipeline=train_pipeline,
#     ),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='test/images',
#         ann_dir='test/ann_masks',
#         split='test.txt',
#         pipeline=test_pipeline,
#     ),
#     test=dict(  # same as val
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='test/images',
#         ann_dir='test/ann_masks',
#         split='test.txt',
#         pipeline=test_pipeline,
#     )
# )

# # 5. model head: just override num_classes
# model = dict(
#     data_preprocessor=data_preprocessor,
#     decode_head=dict(num_classes=num_classes),
#     auxiliary_head=dict(num_classes=num_classes),  # if used
# )

# # the rest (optimizer, scheduler, dataloaders) can stay as in your dummy
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=6e-05, betas=(0.9,0.999), weight_decay=0.01
#     ),
#     paramwise_cfg=dict(
#         custom_keys={
#             'pos_block': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.),
#             'head': dict(lr_mult=10.)
#         }
#     )
# )

# param_scheduler = [
#     dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=160000, by_epoch=False),
# ]

# train_dataloader = dict(batch_size=2, num_workers=2)
# val_dataloader   = dict(batch_size=1, num_workers=4)
# test_dataloader  = val_dataloader

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/my_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]


dataset_type = 'YourDataset'  # replace with your dataset class name
data_root    = '/home2/yasharora120/segmentation_benchmark/data/food'

img_suffix  = '.jpg'
seg_map_suffix = '.png'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53],
                          std=[58.395, 57.12, 57.375]),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=train_pipeline),
    test=dict(  # optional
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=train_pipeline),
)

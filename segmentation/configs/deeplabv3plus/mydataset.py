# configs/deeplabv3plus/my_dataset.py

# 1. Inherit base configs: model, dataset, runtime, schedule
_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py',  # DeepLabV3+ primitive :contentReference[oaicite:6]{index=6}
    '../_base_/datasets/my_dataset.py',     # your custom dataset meta :contentReference[oaicite:7]{index=7}
    '../_base_/default_runtime.py',         # logging & checkpoint defaults :contentReference[oaicite:8]{index=8}
    '../_base_/schedules/schedule_40k.py'   # 40k-iteration lr schedule :contentReference[oaicite:9]{index=9}
]

# 2. Model adjustments: match your N classes
model = dict(
    decode_head=dict(
        num_classes= 61  # replace N with your number of categories :contentReference[oaicite:10]{index=10}
    ),
    # If the base config has an auxiliary head
    auxiliary_head=dict(
        num_classes= 61
    )
)

# 3. Dataset settings: paths and preprocessing
data = dict(
    samples_per_gpu= 4,  # adjust per your GPU memory
    workers_per_gpu= 2,
    train=dict(
        img_dir='data/my_dataset/img_dir/train',   # training images :contentReference[oaicite:11]{index=11}
        ann_dir='data/my_dataset/ann_dir/train',   # indexed masks
    ),
    val=dict(
        img_dir='data/my_dataset/img_dir/val',     # validation images
        ann_dir='data/my_dataset/ann_dir/val',
    ),
    test=dict(
        img_dir='data/my_dataset/img_dir/val',     # reuse val for testing if needed
        ann_dir='data/my_dataset/ann_dir/val',
    )
)

# 4. (Optional) Evaluation metrics and checkpointing
evaluation = dict(metric='mIoU', interval=4000)  # run eval every 4000 iterations
checkpoint_config = dict(by_epoch=False, interval=4000)

# 5. Logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),  # enable if using TensorBoard
    ]
)

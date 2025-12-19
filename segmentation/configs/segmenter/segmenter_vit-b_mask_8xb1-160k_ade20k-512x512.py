_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    # '../_base_/datasets/ade20k.py',
     '../_base_/datasets/your_dataset_ade.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_6k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=1)
val_dataloader = dict(batch_size=1)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='SegVisualizationHook')
    visualization=dict(
            type='SegVisualizationHook',
            draw=True,                  # draw the overlay
            show=False,                 # don’t pop up a window
            interval=1                  # every 1 val epoch
        )
)
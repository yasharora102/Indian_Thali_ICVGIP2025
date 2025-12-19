_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/your_dataset_big.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]
randomness = dict(seed=268722126)
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

# mmseg/mmseg/datasets/food_dataset.py

from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS

@DATASETS.register_module()
class FoodDataset(CustomDataset):
    # will be loaded from data/food/categories.json
    # but you can hard‐code if you prefer:
    CLASSES = tuple(
        open('data/food/categories.json').read().replace('"','').strip("[]\n ").split(", ")
    )
    # A dummy palette—mmsegmentation only requires it for visualization
    PALETTE = [[i*10 % 256, i*20 % 256, i*30 % 256] for i in range(len(CLASSES))]

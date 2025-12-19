from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import pandas as pd

@DATASETS.register_module()
class YourDataset(BaseSegDataset):
    METAINFO = {
        'classes': [
            'background', 'Cabbage', 'Cabbage-curry', 'aloo-curry', 'aloo-fry', 'aloo-gobi-fry',
            'bagara-chawal', 'baingan-curry', 'banana-chips', 'beet-root', 'beetroot-porial',
            'besi-bele-bath', 'bitter-gourd-curry', 'bitter-gourd-non-spicy-curry',
            'bitter-gourd-spicy-curry', 'bottle-gourd', 'bottle-grourd-curry', 'break-fast',
            'cucumber-dal', 'curd', 'curd-raitha', 'dal', 'dal-makhani', 'drink',
            'fresh-chutney', 'fruit-of-the-day', 'green-moong-dal', 'green-salad', 'jeera-pulao',
            'kabuli-chana', 'kadai-paneer', 'khadi-pakoda', 'lady-finger',
            'lady-finger-spicy-curry', 'live-palak-roti-with-ghee', 'live-roti-with-ghee',
            'malai-kofta-curry', 'meal-maker-curry', 'mirchi-ka-salan', 'moong-dal',
            'non-spicy-curry', 'non-spicy-curry2', 'non-spicy-dal', 'papad', 'pappu-charu',
            'plain-rice', 'potato', 'pulihora', 'rajma', 'rasam', 'seviyaan', 'soop',
            'spicy-dal', 'spl-laddu', 'thotakura-pappu', 'tindora', 'tomato-drumstic-curry',
            'turaiya', 'veg-dum-biryani', 'sewaiyan', 'namkeen'
        ],
        'palette': [
            [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
            [255, 0, 255], [153, 0, 153], [255, 165, 0], [153, 153, 0], [0, 153, 153],
            [255, 69, 0], [0, 102, 0], [70, 128, 180], [220, 20, 60], [255, 140, 0],
            [0, 205, 205], [139, 35, 229], [157, 205, 50], [255, 99, 69], [0, 250, 154],
            [71, 205, 202], [140, 107, 209], [65, 63, 69], [255, 215, 0], [219, 107, 204],
            [60, 179, 110], [210, 105, 30], [255, 20, 140], [0, 255, 128], [102, 176, 230],
            [255, 0, 128], [31, 179, 170], [176, 0, 207], [50, 180, 75], [0, 186, 255],
            [180, 50, 120], [128, 255, 0], [200, 70, 250], [240, 230, 140], [177, 200, 221],
            [255, 224, 195], [95, 158, 160], [255, 128, 80], [10, 200, 90], [90, 50, 220],
            [216, 165, 30], [255, 216, 176], [100, 137, 33], [205, 91, 91], [255, 193, 203],
            [177, 222, 230], [100, 240, 200], [71, 54, 139], [60, 180, 220], [0, 139, 139],
            [180, 128, 10], [139, 0, 139], [20, 20, 107], [0, 0, 139], [255, 105, 180]
        ]
    }

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg',
                         seg_map_suffix='_colored_combined_mask.png',
                         reduce_zero_label=False, 
                         **kwargs)

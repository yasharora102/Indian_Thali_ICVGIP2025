# from mmseg.registry import DATASETS
# from .basesegdataset import BaseSegDataset
# import pandas as pd

# @DATASETS.register_module()
# class YourDataset_BIG(BaseSegDataset):
#     METAINFO = {
#         'classes': ['background', 'Aloo Dry fry', 'Avakaya Muddha Papu Rice', 'Baby-Corn & Capsicum-Dry', 'Chakar-Pongal', 'Chole-Masala', 'Cucumber-Raitha', 'Dal-Fry', 'Drink', 'Gobi Masala Curry', 'Moong-Beans-Curry', 'Muskmelon', 'Pappu-Charu', 'Tomato Rasam', 'Vankaya-Ali-Karam', 'Veg-Biriyani', 'aloo-curry', 'banana', 'curd', 'dal', 'fresh-chutney', 'green-salad', 'ivy-dal-gourd-curry', 'khichdi', 'lemon-rice', 'live-roti-with-ghee', 'non-spicy-curry-bottle-gourd', 'non-spicy-dal', 'plain-rice', 'rasam', 'sambar-rice', 'watermelon'],
#         'palette': [(0, 0, 0), (251, 228, 90), (207, 4, 127), (125, 93, 164), (104, 34, 134), (68, 230, 65), (133, 191, 210), (44, 80, 228), (32, 160, 128), (70, 216, 80), (178, 80, 80), (20, 221, 196), (132, 197, 240), (114, 238, 131), (254, 3, 102), (228, 137, 198), (92, 204, 242), (32, 224, 192), (249, 139, 221), (243, 84, 116), (250, 250, 55), (168, 221, 157), (102, 255, 102), (153, 200, 59), (83, 118, 69), (238, 182, 101), (255, 0, 124), (69, 115, 79), (238, 65, 182), (45, 177, 225), (230, 43, 227), (80, 224, 192)]

#     }

#     def __init__(self, **kwargs):
#         super().__init__(img_suffix='_leftImg8bit.jpg',
#                          seg_map_suffix='_gtFine_labelIds.png',
#                          reduce_zero_label=False, 
#                          **kwargs)



from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import pandas as pd

# @DATASETS.register_module()
# class YourDataset_BIG(BaseSegDataset):
#     METAINFO = {
#         'classes': ["background",
# "Aloo Dry fry",
# "Avakaya Muddha Papu Rice",
# "Baby-Corn & Capsicum-Dry",
# "Cabbage Pakodi",
# "Cabbage fry",
# "Capsicum Paneer Curry",
# "Chakar-Pongal",
# "Chole-Masala",
# "Cluster Beans Curry",
# "Cucumber-Raitha",
# "Dal-Fry",
# "Gobi Masala Curry",
# "Gutti Vankaya Curry",
# "Jeera Rice",
# "Leaf Dal",
# "Mixed Curry",
# "ivy-dal-gourd-curry",
# "Muskmelon",
# "pepper-rasam",
# "Rajma",
# "Rasgulla",
# "Sambar",
# "Tomato Rasam",
# "Vankaya-Ali-Karam",
# "Veg-Biriyani",
# "aloo-curry",
# "curd",
# "dal",
# "fresh-chutney",
# "green-salad",
# "Moong-Beans-Curry",
# "khichdi",
# "lemon-rice",
# "live-roti-with-ghee",
# "non-spicy-curry-bottle-gourd",
# "non-spicy-dal",
# "papad",
# "plain-rice",
# "watermelon"],
#         'palette': 
#             [
#     (0, 0, 0),            # background
#     (255, 99, 71),        # Aloo Dry fry
#     (255, 140, 0),        # Avakaya Muddha Papu Rice
#     (255, 215, 0),        # Baby-Corn & Capsicum-Dry
#     (154, 205, 50),       # Cabbage Pakodi
#     (85, 107, 47),        # Cabbage fry
#     (50, 205, 50),        # Capsicum Paneer Curry
#     (0, 128, 0),          # Chakar-Pongal
#     (46, 139, 87),        # Chole-Masala
#     (102, 205, 170),      # Cluster Beans Curry
#     (64, 224, 208),       # Cucumber-Raitha
#     (72, 209, 204),       # Dal-Fry
#     (70, 130, 180),       # Gobi Masala Curry
#     (30, 144, 255),       # Gutti Vankaya Curry
#     (100, 149, 237),      # Jeera Rice
#     (123, 104, 238),      # Leaf Dal
#     (138, 43, 226),       # Mixed Curry
#     (147, 112, 219),      # ivy-dal-gourd-curry
#     (186, 85, 211),       # Muskmelon
#     (218, 112, 214),      # Pappu-Charu
#     (255, 105, 180),      # Rajma
#     (255, 20, 147),       # Rasgulla
#     (219, 112, 147),      # Sambar
#     (205, 92, 92),        # Tomato Rasam
#     (240, 128, 128),      # Vankaya-Ali-Karam
#     (250, 128, 114),      # Veg-Biriyani
#     (233, 150, 122),      # aloo-curry
#     (210, 105, 30),       # curd
#     (160, 82, 45),        # dal
#     (255, 160, 122),      # fresh-chutney
#     (255, 228, 181),      # green-salad
#     (189, 183, 107),      # Moong-Beans-Curry
#     (255, 239, 213),      # khichdi
#     (255, 222, 173),      # lemon-rice
#     (240, 230, 140),      # live-roti-with-ghee
#     (238, 232, 170),      # non-spicy-curry-bottle-gourd
#     (176, 224, 230),      # non-spicy-dal
#     (135, 206, 250),      # papad
#     (70, 130, 180),       # plain-rice
#     (250, 250, 210),      # watermelon
# ]

#     }

#     def __init__(self, **kwargs):
#         super().__init__(img_suffix='_leftImg8bit.jpg',
#                          seg_map_suffix='_gtFine_labelIds.png',
#                          reduce_zero_label=False, 
#                          **kwargs)



# @DATASETS.register_module()
# class YourDataset_BIG(BaseSegDataset):
#     METAINFO = {
#         'classes': ["background",
# "Aloo Dry fry",
# "Avakaya Muddha Papu Rice",
# "Baby-Corn & Capsicum-Dry",
# "Cabbage Pakodi",
# "Cabbage fry",
# "Capsicum Paneer Curry",
# "Chakar-Pongal",
# "Chole-Masala",
# "Cluster Beans Curry",
# "Cucumber-Raitha",
# "Dal-Fry",
# "Gobi Masala Curry",
# "Gutti Vankaya Curry",
# "Jeera Rice",
# "Leaf Dal",
# "Mixed Curry",
# "ivy-dal-gourd-curry",
# "Muskmelon",
# "Pappu-Charu",
# "Rajma",
# "Rasgulla",
# "Sambar",
# "Tomato Rasam",
# "Vankaya-Ali-Karam",
# "Veg-Biriyani",
# "aloo-curry",
# "curd",
# "dal",
# "fresh-chutney",
# "green-salad",
# "Moong-Beans-Curry",
# "khichdi",
# "lemon-rice",
# "live-roti-with-ghee",
# "non-spicy-curry-bottle-gourd",
# "non-spicy-dal",
# "papad",
# "plain-rice",
# "watermelon",
# "Aloo-Fry",
# "Banana",
# "Mix-Fruit",
# "Non-Spicy-Baby-Corn & Capsicum-Dry",
# "Sweet",
# "Tomato-Rice",
# "fried-papad-rings",
# "gravy",
# "ivy-gourd-fry",
# "mango-pickle",
# "papad-chat",
# "pepper-rasam",
# "pineapple",
# "corn-fry",
# "horse-gram-curry",
# "paneer-curry",
# "semiya"],
#         'palette': 
#             [
#     (0, 0, 0),             # background
#     (139, 0, 0),           # Aloo Dry fry
#     (255, 69, 0),          # Avakaya Muddha Papu Rice
#     (255, 140, 0),         # Baby-Corn & Capsicum-Dry
#     (255, 215, 0),         # Cabbage Pakodi
#     (173, 255, 47),        # Cabbage fry
#     (124, 252, 0),         # Capsicum Paneer Curry
#     (0, 128, 0),           # Chakar-Pongal
#     (0, 255, 127),         # Chole-Masala
#     (0, 206, 209),         # Cluster Beans Curry
#     (0, 191, 255),         # Cucumber-Raitha
#     (30, 144, 255),        # Dal-Fry
#     (65, 105, 225),        # Gobi Masala Curry
#     (75, 0, 130),          # Gutti Vankaya Curry
#     (138, 43, 226),        # Jeera Rice
#     (147, 112, 219),       # Leaf Dal
#     (199, 21, 133),        # Mixed Curry
#     (255, 20, 147),        # ivy-dal-gourd-curry
#     (255, 99, 71),         # Muskmelon
#     (255, 160, 122),       # Pappu-Charu
#     (240, 230, 140),       # Rajma
#     (255, 250, 205),       # Rasgulla
#     (245, 222, 179),       # Sambar
#     (210, 180, 140),       # Tomato Rasam
#     (188, 143, 143),       # Vankaya-Ali-Karam
#     (169, 169, 169),       # Veg-Biriyani
#     (105, 105, 105),       # aloo-curry
#     (112, 128, 144),       # curd
#     (47, 79, 79),          # dal
#     (0, 100, 0),           # fresh-chutney
#     (34, 139, 34),         # green-salad
#     (50, 205, 50),         # Moong-Beans-Curry
#     (173, 216, 230),       # khichdi
#     (135, 206, 235),       # lemon-rice
#     (240, 248, 255),       # live-roti-with-ghee
#     (176, 196, 222),       # non-spicy-curry-bottle-gourd
#     (100, 149, 237),       # non-spicy-dal
#     (175, 238, 238),       # papad
#     (127, 255, 212),       # plain-rice
#     (64, 224, 208),        # watermelon
#     (72, 209, 204),        # Aloo-Fry
#     (84, 255, 159),        # Banana
#     (127, 255, 0),         # Mix-Fruit
#     (204, 255, 0),         # Non-Spicy-Baby-Corn & Capsicum-Dry
#     (255, 255, 0),         # Sweet
#     (255, 239, 213),       # Tomato-Rice
#     (255, 228, 196),       # fried-papad-rings
#     (255, 228, 181),       # gravy
#     (255, 218, 185),       # ivy-gourd-fry
#     (255, 192, 203),       # mango-pickle
#     (255, 182, 193),       # papad-chat
#     (255, 105, 180),       # pepper-rasam
#     (255, 0, 255),         # pineapple
#     (218, 112, 214),       # corn-fry
#     (238, 130, 238),       # horse-gram-curry
#     (153, 50, 204),        # paneer-curry
#     (128, 0, 128)          # semiya
# ]

#     }
@DATASETS.register_module()
class YourDataset_BIG(BaseSegDataset):
    METAINFO = {
        'classes': [
            # "background",
            # "Aloo Dry fry",
            # "Avakaya Muddha Papu Rice",
            # "Baby-Corn & Capsicum-Dry",
            # "Cabbage Pakodi",
            # "Cabbage fry",
            # "Capsicum Paneer Curry",
            # "Chakar-Pongal",
            # "Chole-Masala",
            # "Cluster Beans Curry",
            # "Cucumber-Raitha",
            # "Gobi Masala Curry",
            # "Gutti Vankaya Curry",
            # "Jeera Rice",
            # "Mixed Curry",
            # "Muskmelon",
            # "Rajma",
            # "Rasgulla",
            # "Sambar",
            # "Tomato Rasam",
            # "Vankaya-Ali-Karam",
            # "Veg-Biriyani",
            # "aloo-curry",
            # "curd",
            # "dal",
            # "fresh-chutney",
            # "green-salad",
            # "Moong-Beans-Curry",
            # "khichdi",
            # "lemon-rice",
            # "live-roti-with-ghee",
            # "non-spicy-curry-bottle-gourd",
            # "papad",
            # "plain-rice",
            # "watermelon",
            # "Aloo-Fry",
            # "Banana",
            # "Mix-Fruit",
            # "Non-Spicy-Baby-Corn & Capsicum-Dry",
            # "Sweet",
            # "Tomato-Rice",
            # "fried-papad-rings",
            # "gravy",
            # "ivy-gourd-fry",
            # "mango-pickle",
            # "papad-chat",
            # "pepper-rasam",
            # "pineapple",
            # "corn-fry",
            # "paneer-curry",
            # "semiya"
"background",
"Bottle-gourd-curry",
"aloo-capsicum",
"aloo-curry",
"aloo-fry",
"beans-curry",
"beetroot-kobari",
"beetroot-poriyal",
"bisi-bele-bath",
"boondi",
"cabbage-dry",
"channa-brinjal",
"chicken-dum-biryani",
"chutney",
"curd",
"dondakaya-fry",
"kakarakaya-fry",
"kofta-curry",
"leaf-dal",
"mango-pickle",
"masoor-dal",
"mirchi-ka-salan",
"muddha-pappu",
"non-spicy-curry",
"non-spicy-dal",
"pachi-pulusu",
"papad",
"payasam",
"phulka",
"raita",
"rajma",
"rasam",
"salad",
"sambar",
"steamed-rice",
"tomato-pappu",
"veg-dum-briyani",
"veg-pulao",
"Watermelon",
"Papaya",
"Banana",
"Muskmelon"       
],
        'palette': 
            [
      (0, 0, 0),          # background
    (224, 75, 126),     # Bottle-gourd-curry
    (24, 219, 114),     # aloo-capsicum
    (93, 201, 154),     # aloo-curry
    (153, 107, 191),    # aloo-fry
    (208, 126, 211),    # beans-curry
    (22, 194, 25),      # beetroot-kobari
    (45, 19, 210),      # beetroot-poriyal
    (252, 9, 21),       # bisi-bele-bath
    (100, 185, 137),    # boondi
    (255, 130, 241),    # cabbage-dry
    (185, 237, 241),    # channa-brinjal
    (253, 76, 231),     # chicken-dum-biryani
    (254, 189, 11),     # chutney
    (169, 136, 172),    # curd
    (142, 11, 23),      # dondakaya-fry
    (184, 170, 72),     # kakarakaya-fry
    (251, 196, 226),    # kofta-curry
    (208, 223, 222),    # leaf-dal
    (219, 105, 14),     # mango-pickle
    (144, 215, 203),    # masoor-dal
    (169, 102, 109),    # mirchi-ka-salan
    (109, 173, 14),     # muddha-pappu
    (183, 239, 114),    # non-spicy-curry
    (90, 73, 227),      # non-spicy-dal
    (157, 139, 19),     # pachi-pulusu
    (225, 237, 153),    # papad
    (102, 151, 88),     # payasam
    (186, 139, 231),    # phulka
    (139, 112, 128),    # raita
    (223, 17, 149),     # rajma
    (198, 232, 15),     # rasam
    (24, 222, 177),     # salad
    (199, 184, 126),    # sambar
    (212, 185, 228),    # steamed-rice
    (180, 238, 21),     # tomato-pappu
    (230, 93, 223),     # veg-dum-briyani
    (215, 184, 15),     # veg-pulao
    (216, 231, 200),    # Watermelon
    (225, 59, 16),      # Papaya
    (230, 8, 234),      # Banana
    (145, 198, 114)     # Muskmelon
    # (204, 102, 255),     # fried-papad-rings
    # (255, 153, 51),      # gravy
    # (153, 51, 255),      # ivy-gourd-fry
    # (51, 255, 153),      # mango-pickle
    # (153, 255, 51),      # papad-chat
    # (51, 153, 255),      # pepper-rasam
    # (255, 51, 153),      # pineapple
    # (51, 255, 51),       # corn-fry
    # (51, 51, 255),       # paneer-curry
    # (153, 153, 153)      # semiya
]

    }
    def __init__(self, **kwargs):
        super().__init__(img_suffix='_leftImg8bit.jpg',
                         seg_map_suffix='_gtFine_labelIds.png',
                         reduce_zero_label=False, 
                         **kwargs)




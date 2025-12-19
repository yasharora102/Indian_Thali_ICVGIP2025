## Prototype Download
Download the prototype embeddings from the Releases section. [here](https://github.com/yasharora102/Indian_Thali/releases/download/models/FINAL_WED_NEW_PLATES_PE.zip)

Download prototype data for WED under the Releases Section. [here](https://github.com/yasharora102/Indian_Thali/releases/download/models/proto_WED.zip)



## Custom Data
To use your own custom data with the Food Scanner, follow these steps:

Put all prototypes as the following structure:
```bash

./data/prototypes
├── 20250512
│   ├── beetroot-poriyal
│   │   ├── beetroot-poriyal_1.jpg
│   │   ├── beetroot-poriyal_2.jpg
│   │   └── beetroot-poriyal_3.jpg
│   ├── cabbage-dry
│   │   ├── cabbage-dry_1.jpg
│   │   ├── cabbage-dry_2.jpg
│   │   └── cabbage-dry_3.jpg
... more food categories
```

Then, run the following command to generate the prototype file after updating the path in the script:

```bash
EMBEDDINGS_DIR = f"/OUT_PATH"  #ouput dir
PROTO_DIR = "INPUT" #input dir
```

Run the script:
```bash
python generate_embeddings.py
```

Set the correct paths in `app.py`:
```python
CFG_PATH     = "experiment_core_patches_50.yaml_PATH"  # path to config file
IDXMAP_PATH  = "PATH_TO_index_map_weight.json"          # {"0":"background","1":"ClassA",...}
PROTOS_DIR   = f"/OUT_PATH"             # prototypes/YYYYMMDD/*.
WEIGHT_CKPT  = "PATH_TO_WEIGHT_CKPT"
```

Now run the Food Scanner app:
```bash
uvicorn app:app --host 0.0.0.0 --port 8003
```

## Limitations
- The Weight Estimation model is trained on a specific set of Indian food items. It will not perform well on food items outside this set.
- We are working to to make a mapping script to map new food items to the existing classes.

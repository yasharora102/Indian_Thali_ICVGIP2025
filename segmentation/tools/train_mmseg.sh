#!/usr/bin/env bash
# from your_project/mmseg/
export PYTHONPATH=$(pwd):$PYTHONPATH

python tools/train.py \
  configs/food/food_segformer_1024x1024.py \
  --work-dir work_dirs/food_segformer

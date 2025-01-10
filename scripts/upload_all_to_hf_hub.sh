#!/bin/bash

path="src/postprocessing"
is_tuned="untuned"
strategy="ddp"
precision=32
batch_size=16
model_detail="multimodal-transformer"

python $path/upload_all_to_hf_hub.py --config-name=multimodal.yaml \
    is_tuned=$is_tuned \
    strategy=$strategy \
    precision=$precision \
    batch_size=$batch_size \
    model_detail=$model_detail
#!/bin/bash

path="src/postprocessing"
is_tuned="untuned"
strategy="ddp"
image_upload_user="microsoft"
image_model_type="dit-large-finetuned-rvlcdip"
text_upload_user="klue"
text_model_type="roberta-large"
precision=32
batch_size=16
epoch=10

python $path/prepare_upload.py --config-name=huggingface.yaml \
    is_tuned=$is_tuned \
    strategy=$strategy \
    image_upload_user=$image_upload_user \
    image_model_type=$image_model_type \
    text_upload_user=$text_upload_user \
    text_model_type=$text_model_type \
    precision=$precision \
    batch_size=$batch_size \
    epoch=$epoch

#!/bin/bash

is_tuned="untuned"
strategy="ddp"
image_upload_user="microsoft"
image_model_type="dit-large-finetuned-rvlcdip"
text_upload_user="klue"
text_model_type="roberta-large"
precision=32
batch_size=16
epochs="9 10"

for epoch in $epochs
do
    python main.py --config-name=multimodal.yaml \
        mode=predict \
        is_tuned=$is_tuned \
        strategy=$strategy \
        image_upload_user=$image_upload_user \
        image_model_type=$image_model_type \
        text_upload_user=$text_upload_user \
        text_model_type=$text_model_type \
        precision=$precision \
        batch_size=$batch_size
done

modality="text"
is_tuned="untuned"
strategy="ddp"
upload_user="klue"
model_type="roberta-large"
precision=32
batch_size=24
epochs="9 10"

for epoch in $epochs
do
    python main.py --config-name=huggingface.yaml \
        mode=predict \
        modality=$modality \
        is_tuned=$is_tuned \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        precision=$precision \
        batch_size=$batch_size
done

modality="image"
is_tuned="untuned"
strategy="ddp"
upload_user="microsoft"
model_type="dit-large-finetuned-rvlcdip"
precision=32
batch_size=24
epochs="9 10"

for epoch in $epochs
do
    python main.py --config-name=huggingface.yaml \
        mode=predict \
        modality=$modality \
        is_tuned=$is_tuned \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        precision=$precision \
        batch_size=$batch_size
done

modality="multimodal"
is_tuned="untuned"
strategy="ddp"
upload_user="nielsr"
model_type="layoutlmv3-finetuned-cord"
precision=32
batch_size=24
epochs="9 10"

for epoch in $epochs
do
    python main.py --config-name=huggingface.yaml \
        mode=predict \
        modality=$modality \
        is_tuned=$is_tuned \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        precision=$precision \
        batch_size=$batch_size
done

is_tuned="untuned"
strategy="ddp"
model_type="efficientnet_b0.ra_in1k"
pretrained="pretrained"
precision=32
batch_size=24
epochs="9 10"

for epoch in $epochs
do
    python main.py --config-name=timm.yaml \
        mode=predict \
        is_tuned=$is_tuned \
        strategy=$strategy \
        model_type=$model_type \
        pretrained=$pretrained \
        precision=$precision \
        batch_size=$batch_size
done

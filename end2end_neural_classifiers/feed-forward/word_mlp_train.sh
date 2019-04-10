#!/bin/bash

#python -c 'import torch;print("Pytorch version is "+torch.__version__)'
#Pytorch version is 0.4.1

input_dir="2class_balanced_ptfiles"
# or input_dir = 5class_ptfiles

model_dir="2class_balanced_models"
# or model_dir="5class_models"

result_dir="2class_balanced_preds_and_labels"
# or result_dir="5class_preds_and_labels"

rm -r ${model_dir}
rm -r ${result_dir}

mkdir ${model_dir}
mkdir ${result_dir}

for i in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0
    python -u ./nwa/nwa.py \
           -data ../dataset/k-folds-normalized/${input_dir}/fold${i}.multidata.en-fr.pt \
           -save_path ./${model_dir}/models \
           -save_every 1 \
           -validate_every 1 \
           -sample_every 5 \
           -learning_rate 0.0001 \
           -epoch 200 \
           -embedding_size 10 \
           -hidden_size 10 \
           -batch_size 20 \
           -dropout 0.1 \
           -weight_decay 0.0 \
           -clip 0.0 \
           -pretrained_embedding_src ../dataset/en-fr-corpus/fasttext-normalized/embedding_en.pt \
           -pretrained_embedding_tgt ../dataset/en-fr-corpus/fasttext-normalized/embedding_fr.pt \
           -log ./${model_dir}/models
           # -cuda \   remove this option to run on CPU

    mkdir ${result_dir}/fold_${i}
    cp ./${model_dir}/models_best_all_preds.txt ./${result_dir}/fold_${i}
    cp ./${model_dir}/models_best_all_labels.txt ./${result_dir}/fold_${i}
done





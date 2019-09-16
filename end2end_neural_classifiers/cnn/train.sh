#!/bin/bash

# sh train.sh 1/2/3/4/5 <- the number of fold being cross-validating

#python --version
python -c 'import torch;print("Pytorch version is "+torch.__version__)'

# You should change here according to the dataset
input_dir="2class_balanced_ptfiles_char"
#input_dir="5class_ptfiles_char"

model_dir="2class_balanced_models"
#model_dir="5class_models"

result_dir="2class_balanced_preds_and_labels"
#result_dir="5class_preds_and_labels"

rm -r ${model_dir}
rm -r ${result_dir}

mkdir ${model_dir}
mkdir ${result_dir}

#CUDA_VISIBLE_DEVICES=1
# TODO: the number of epoch should be 200

for i in 1 2 3 4 5
do
    python -u ./nwa/nwa.py \
           -data ../dataset/k-folds-normalized/${input_dir}/fold${i}.multidata.char.fr-en.pt \
           -save_path ./${model_dir}/models \
           -save_every 1 \
           -validate_every 1 \
           -sample_every 5 \
           -learning_rate 0.0001 \
           -epoch 20 \
           -embedding_size 10 \
           -hidden_size 10 \
           -batch_size 20 \
           -dropout 0.1 \
           -weight_decay 0.0 \
           -clip 0.0 \
           -pretrained_embedding_src ../dataset/polyglot_embeddings/embedding_src.pt \
           -pretrained_embedding_tgt ../dataset/polyglot_embeddings/embedding_tgt.pt \
           -log ./${model_dir}/models
           # -cuda \   remove cuda to run on CPU

    mkdir ${result_dir}/fold_${i}
    cp ./${model_dir}/models_best_all_preds.txt ./${result_dir}/fold_${i}
    cp ./${model_dir}/models_best_all_labels.txt ./${result_dir}/fold_${i}
done

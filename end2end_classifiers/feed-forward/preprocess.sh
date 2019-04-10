#!/bin/bash

# input files are produced by dataset/create-CV.py
input_dir="cross_validation_2class_balanced"
# or name_dir="cross_validation_5class"

output_dir="2class_balanced_ptfiles"
# or output_dir = 5class_ptfiles

OUTDIR="../dataset/k-folds-normalized/${output_dir}"

if [ ! -d "${OUTDIR}" ]; then
    mkdir ${OUTDIR}
fi

for i in 1 2 3 4 5
do
   python -u ./nwa/preprocess.py \
        -train_src ../dataset/k-folds-normalized/${input_dir}/source_train_${i}.txt \
        -train_tgt ../dataset/k-folds-normalized/${input_dir}/target_train_${i}.txt \
        -train_labels ../dataset/k-folds-normalized/${input_dir}/label_train_${i}.txt \
        -valid_src ../dataset/k-folds-normalized/${input_dir}/source_test_${i}.txt \
        -valid_tgt ../dataset/k-folds-normalized/${input_dir}/target_test_${i}.txt \
        -valid_labels ../dataset/k-folds-normalized/${input_dir}/label_test_${i}.txt \
        -vocab ../dataset/en-fr-corpus/fasttext-normalized/predefined_dicts.fr-en.pt \
        -min_word_count 0 \
        -save_data ../dataset/k-folds-normalized/${output_dir}/fold${i}.multidata.en-fr.pt
done





    
    

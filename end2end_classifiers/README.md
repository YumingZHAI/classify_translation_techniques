## Preparation work
 
Preprocessing steps for corpus: lowercasing, correcting minor spelling errors. For end2end neural architectures, we also normalized the clitic forms to complete words (e.g. 're -> are), and normalized digits to letter form (e.g. 42 -> four two).

We conducted experiments with balanced binary classes dataset and five-classes dataset. 

1. **Prepare dataset**

`dataset`/2class_balanced.csv, `dataset`/5classes.csv <br/>
`dataset/create-CV.py` => `k-folds-normalized`/cross_validation dataset 

2. **Pretrain fasttext word embeddings**

generate word embeddings from normalized corpus (3M En words, ted talks) <br/> 
`cd feed-forward` <br/> 
`sh fasttext_word_representation.sh`




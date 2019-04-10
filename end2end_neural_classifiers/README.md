## Preparation work
 
Preprocessing steps for corpus: lowercasing, correcting minor spelling errors. For end2end neural architectures, we also normalized the clitic forms to complete words (e.g. 're -> are), and normalized digits to letter form (e.g. 42 -> four two).

We conducted experiments with balanced binary classes dataset and five-classes dataset. 

1. **Prepare dataset**

`dataset`/2class_balanced.csv, `dataset`/5classes.csv <br/>
`python dataset/create-CV.py` => `k-folds-normalized`/cross_validation dataset 

2. **Pretrain fasttext word embeddings**

Generate word embeddings from normalized corpus (3M words in English and French, corpus of Ted Talks) <br/> 
`sh feed-forward/fasttext_word_representation.sh`

Extract fasttext word embedding tensors from embeddings text file and build the predefined dictionary for the data. 

`python feedforward/nwa/fasttext_embedding_tensor.py`

<!-- how character embeddings are obtained  -->

3. **Save dataset to serialized files**

`sh feed-forward/preprocess.sh`

4. **Train different architectures**

<!-- ### questions on CNN architecture:
the code to build alignment matrix 
what does adaptive pooling do 

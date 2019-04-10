## Preparation work
 
Preprocessing steps for corpus: lowercasing, correcting minor spelling errors. For end2end neural architectures, we also normalized the clitic forms to complete words (e.g. 're -> are), and normalized digits to letter form (e.g. 42 -> four two).

We conducted experiments with balanced binary classes dataset and five-classes dataset. 

1. **Prepare dataset**

input: `dataset`/2class_balanced.csv, `dataset`/5classes.csv <br/>
script: `python dataset/create-CV.py` <br/>
output: `k-folds-normalized`/cross_validation dataset (csv format)

2. **Pretrain FastText word embeddings**

Generate word embeddings from normalized corpus (3M words in English and French, corpus of Ted Talks) <br/> 
script: `sh feed-forward/fasttext_word_representation.sh` <br/> 
output position: `dataset/en-fr-corpus/fasttext-normalized/`

Extract fasttext word embedding tensors from embeddings text file and build the predefined dictionary for the data. <br/>     
script: `python feedforward/fasttext_embedding_tensor.py` <br/>     
output: `dataset/en-fr-corpus/fasttext-normalized/`predefined_dicts.fr-en.pt, embedding_{en,fr}.pt

3. **Extract character embeddings from [Polyglot word embeddings](https://sites.google.com/site/rmyeid/projects/polyglot)**

<!-- how character embeddings are obtained  -->

4. **Save dataset to serialized files**

script: `sh feed-forward/preprocess.sh` <br/>
output: dataset/k-folds-normalized/${output_dir}/fold${i}.multidata.en-fr.pt

## Train different classifiers

- word embedding + MLP classifier: <br/>
`sh feedforward/word_mlp_train.sh`    

- character embedding, word alignment matrix, CNN classifier 

<!-- 
when do 5-class clf instead of 2-class:
change nwa/nwa.py: 


-->


<!-- ### questions on CNN architecture:
the code to build alignment matrix 
what does adaptive pooling do 
masked_cross_entropy? 

files not yet uploaded
dataset/en-fr-corpus/fasttext-normalized/*  1.73G
my pickled files 
-->


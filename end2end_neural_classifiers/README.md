## Preparation work
 
Corpus preprocessing steps: lowercasing and correcting minor spelling errors.
We did this correction according to a list of errors noted down by annotators, thus we can keep the original TED Talks corpus intact. 

For end2end neural architectures, we also normalized the clitic forms to complete words (e.g. 're -> are), and normalized digits to their letter form (e.g. 42 -> four two (not forty-two)).

We conducted experiments with a balanced dataset of binary classes and a dataset of five classes. 

1. **Split dataset to five folds (already supplied here)**

input: `dataset`/2class_balanced.csv, `dataset`/5classes.csv <br/>
script: `python dataset/create-CV.py` <br/>
output: `dataset/k-folds-normalized`/cross_validation_2class_balanced/ or cross_validation_5class/

2. **Pretrain FastText word embeddings**

We used fastText-0.2.0 in this step.

Generate word embeddings from a parallel English-French corpus, with the same preprocessing as cited above. <br/> 
script: `sh fasttext_word_representation.sh` <br/> 
output: `dataset/en-fr-corpus/fasttext-normalized`/{english,french}-normalized.{bin,vec} <br/> 
(Our corpus and model are not loaded here because of the big size.) <!-- ours: normalized corpus (3M words in English and French, corpus of TED Talks) <br/>  -->

Construct fasttext word embedding tensors from embeddings text file and build the predefined dictionary for the data. <br/> 
script: `python fasttext_embedding_tensor.py` <br/> 
output: <br/>
`dataset/en-fr-corpus/fasttext-normalized/`predefined_dicts.fr-en.pt <br/>
`dataset/en-fr-corpus/fasttext-normalized/`embedding_{en,fr}.pt

3. **Extract character embeddings from [Polyglot word embeddings](https://sites.google.com/site/rmyeid/projects/polyglot)**

First download `polyglot-en.pkl, polyglot-fr.pkl` from the site above, put them under `dataset/polyglot_embeddings/`<br/>
script: `python extract_polyglot_embedding.py` (Please run this script under python 2.7.)  <br/>
output: <br/>
`dataset/polyglot_embeddings/`predefined_dicts.char.fr-en.pt <br/>
`dataset/polyglot_embeddings/`embedding_{src,tgt}.pt <br/>
(You should run the script to generate your own copy.)

4. **Save dataset to serialized files**

script: `sh feed-forward/preprocess.sh` <br/>
output: dataset/k-folds-normalized/${output_dir}/fold${i}.multidata.en-fr.pt

## Train different classifiers

- word embedding based + MLP classifier: <br/>
`sh feedforward/word_mlp_train.sh`

- character embedding based + phrase alignment matrix + CNN classifier 

<!-- 
when do 5-class clf instead of 2-class:
change nwa/nwa.py: 

-->


<!-- ### questions on CNN architecture:
the code to build alignment matrix 
what does adaptive pooling do 
what is masked_cross_entropy? 
the forward function in encoders and classifiers <= train()

files not yet uploaded
dataset/en-fr-corpus/fasttext-normalized/*  1.73G
my pickled files 
polyglot embeddings 
-->


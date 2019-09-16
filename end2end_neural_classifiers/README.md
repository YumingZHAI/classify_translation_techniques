## Preparation work
 
<!-- the minor misspellings are already corrected in the corpus -->
We lowercase all the words in the dataset. We normalize the clitic forms to complete words (e.g. 're -> are), and digits to their letter form (e.g. 42 -> four two (not forty-two)).

We conducted experiments with a balanced dataset of binary classes and a dataset of five classes. 

1. **Split dataset to five folds (already supplied here)**

input: `dataset`/2class_balanced.csv, `dataset`/5classes.csv <br/> `cd
dataset/` <br/>  script: `python dataset/create-CV.py` <br/> output:
`dataset/k-folds-normalized/cross_validation_2class_balanced/` or
`cross_validation_5class/`

2. **Pretrain FastText word embeddings**

We used fastText-0.2.0 in this step.

Generate English and French word embeddings from a parallel English-French corpus, with the same preprocessing as cited above. <br/> 
script: `sh fasttext_word_representation.sh` <br/> 
output: `dataset/en-fr-corpus/fasttext-normalized`/{english,french}-normalized.{bin,vec} <br/> 
(Our corpus and model are not loaded here because of the big size.) <!-- ours: normalized corpus (3M words in English and French, corpus of TED Talks) <br/>  -->

Construct fasttext word embedding tensors from embeddings text file and build the predefined dictionary for the data. <br/> 
script: `python fasttext_embedding_tensor.py` <br/> 
output: <br/>
`dataset/en-fr-corpus/fasttext-normalized/`predefined_dicts.fr-en.pt <br/>
`dataset/en-fr-corpus/fasttext-normalized/`embedding_{en,fr}.pt

3. **Extract character embeddings from [Polyglot word embeddings](https://sites.google.com/site/rmyeid/projects/polyglot)**

First download `polyglot-en.pkl, polyglot-fr.pkl` from the site above, put them under `dataset/polyglot_embeddings/`.<br/>
script: `python extract_polyglot_embedding.py` (Please run this script under python 2.7.)  <br/>
output: <br/>
`dataset/polyglot_embeddings/`predefined_dicts.char.fr-en.pt <br/>
`dataset/polyglot_embeddings/`embedding_{src,tgt}.pt <br/>
(You should run the script to generate your own copy of embeddings.)

4. **Save preprocessed dataset to serialized files**

- at word level: <br/>
script: `sh preprocess.sh` (it uses the script `preprocess.py`) <br/>
output: <br/>
`dataset/k-folds-normalized/2class_balanced_ptfiles`/fold{1-5}.multidata.en-fr.pt <br/>
`dataset/k-folds-normalized/5class_ptfiles`/fold{1-5}.multidata.en-fr.pt <br/>
(This script reads five folds of training and test data, convert strings to word indices according to the 
word dictionary generated before. 
And serialize both the bilingual dictionary and processed training, test data. 
You should run it to get your own copy of serialized files.)

- at character level: <br/>
script: `sh preprocess_char.sh` (it uses the script `preprocess_char.py`) <br/>
output: <br/>
`dataset/k-folds-normalized/2class_balanced_ptfiles_char`/fold{1-5}.multidata.en-fr.pt <br/>
`dataset/k-folds-normalized/5class_ptfiles_char`/fold{1-5}.multidata.en-fr.pt <br/>

## Train different classifiers

<!-- now: binary clf; need to check 5-class classifier -->

1. **FastText pretrained word embedding + MLP classifier**

`cd feed-forward/` <br/> 
`sh word_mlp_train.sh` <br/>
Calculate evaluation metrics: <br/> 
`python f_score_cross_valid.py` {2class_balanced,5class} <br/>

2. **Randomly initialized character embedding + MLP classifier** 


3. **Randomly initialized character embedding + word or phrase alignment matrix + CNN classifier** 


<!-- ### questions:
the code to build alignment matrix 
what does adaptive pooling do 
what is masked_cross_entropy? 
the forward function in encoders and classifiers <= train()
-->


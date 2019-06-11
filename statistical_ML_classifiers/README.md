## Preparation work
 
Preprocessing steps for corpus: lowercasing, correcting minor spelling errors. 

The procedures used to generate serialized files (*.p files) under `pickle_res/` are described below. 
<!-- upload all the *.p files in a zipped directory onto github -->

1. **PoS tagging** 

tool: [stanford-corenlp-full-2018-10-05](https://stanfordnlp.github.io/CoreNLP/download.html) <br/>
input: `txt_res/{en,fr}.txt`, `corenlp_properties/{en_pos_lemma, fr_pos}.properties` <br/>
command: `stanford-corenlp-full-2018-10-05/corenlp.sh -props {en_pos_lemma, fr_pos}.properties` <br/>
output: `txt_res/{en,fr}.txt.conll` 

2. **Lemmatisation** 

EN: <br/>
tool: [stanford-corenlp-full-2018-10-05](https://stanfordnlp.github.io/CoreNLP/download.html) <br/>
input: `txt_res/en.txt`, `corenlp_properties/en_pos_lemma.properties` <br/>
command: `stanford-corenlp-full-2018-10-05/corenlp.sh -props en_pos_lemma.properties` <br/>
then script `lemmatisation/extract-lemma-corenlp.py` <br/>
output: `lemmatisation/ENlemme_from_corenlp.txt` <br/>

FR: <br/>
tool: [tree-tagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) (because Stanford Corenlp doesn't provide French lemmatization yet) <br/>
input: `txt_res/fr_noID.txt`  <br/>
then script: `lemmatisation/fr-lemmatize.py` <br/>
output: `lemmatisation/fr_lemma.txt`  <br/>

Generate serialized file: `lemmatisation/pickle_lemma.py` => `pickle_res/{eng,fr}_lemma.p` <br/>

3. **Automatic word alignment**

tool: [Berkeley unsupervised word aligner](https://code.google.com/archive/p/berkeleyaligner/downloads) <br/>

input: an English-French parallel corpus, composed of TED Talks and a filtered part of Paracrawl corpus (in total 1.8M parallel sentence pairs and 41M English tokens)  <br/>
<!-- filter Paracrawl corpus, keep the parallel sentences having at least 10 words at each side 
original corpus: 27M, after this filtering: 19M, we randomly take 2M lines to combine with TED corpus => 2 163 092 lines 
finally clean the combined corpus PC2M_ted.{e,f} by a moses's script: 1 806 680 lines -->
corpus position: `berkeleyaligner_unsupervised/example/train/corpus.{en,fr}` (this corpus will be uploaded) <br/> 
command: `./berkeleyaligner_unsupervised/align example.conf` <br/>
time used: 6h  <br/>
main output: `berkeleyaligner_unsupervised/output/: 1.lexweights, 2.lexweights, training.align` 

Generate serialized file: `preparation/berkeley_wordId_lexTable.py` <br/>
output: `pickle_res/{en,fr}_word_id.p, {en,fr}_entropy.p, berkeley_{forward,reverse}_table.p` <br/>
 

4. **ConceptNet embeddings** 

input: [multilingual-numberbatch-17.06.txt](https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz) <br/>
script: `preparation/CNet_embeddings.py` <br/>
output: `pickle_res/CNet_enfr_embeddings.p`

5. **Dictionary[sentence_ID]=line_index**

script: `preparation/lineID_dico.py` <br/>
output: `pickle_res/line_parseId_dict.p`  

6. **Constituency parsing**




## Extract features

`cd classification/` <br/>
e.g.: `python get-feature.py ../txt_res/1:1.txt` 

## Train and evaluate classifiers by cross-validation 


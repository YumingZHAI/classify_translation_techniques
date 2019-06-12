## Preparation work
 
Preprocessing steps for corpus: lowercasing, correcting minor spelling errors. The dataset present here is ready to be used.  

The procedures used to generate serialized files (*.p files) under `pickle_res/` are described below, you can download them [here](https://www.dropbox.com/s/g2bzit2o6lqabpv/pickle_res.zip?dl=0). 
<!-- upload all the *.p files in a zipped directory onto github -->

1. **PoS tagging** 

tool: [stanford-corenlp-full-2018-10-05](https://stanfordnlp.github.io/CoreNLP/download.html) <br/>
input: `txt_res/{en,fr}.txt`, `corenlp_properties/{en_pos_lemma, fr_pos}.properties` <br/>
command: `stanford-corenlp-full-2018-10-05/corenlp.sh -props {en_pos_lemma, fr_pos}.properties` <br/>
output: `txt_res/{en,fr}.txt.conll` 

2. **Lemmatisation** 

EN: <br/>
tool: stanford-corenlp-full-2018-10-05 <br/>
input: `txt_res/en.txt`, `corenlp_properties/en_pos_lemma.properties` <br/>
command: `stanford-corenlp-full-2018-10-05/corenlp.sh -props en_pos_lemma.properties` <br/>
then script `lemmatisation/extract-lemma-corenlp.py` <br/>
output: `lemmatisation/ENlemme_from_corenlp.txt` <br/>

FR: <br/>
tool: [tree-tagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) (because Stanford Corenlp doesn't provide French lemmatization yet) <br/>
input: `txt_res/fr_noID.txt`  <br/>
then script: `lemmatisation/fr-lemmatize.py` <br/>
output: `lemmatisation/fr_lemma.txt`  <br/>

generate serialized file: `lemmatisation/pickle_lemma.py` => `pickle_res/{eng,fr}_lemma.p` <br/>

3. **Automatic word alignment**

tool: [Berkeley unsupervised word aligner](https://code.google.com/archive/p/berkeleyaligner/downloads) <br/>

input: an English-French parallel corpus, composed of TED Talks and a filtered part of Paracrawl corpus (in total 1.8M parallel sentence pairs and 41M English tokens)  <br/>
<!-- filter Paracrawl corpus, keep the parallel sentences having at least 10 words at each side 
original corpus: 27M, after this filtering: 19M, we randomly take 2M lines to combine with TED corpus => 2 163 092 lines 
finally clean the combined corpus PC2M_ted.{e,f} by a moses's script: 1 806 680 lines -->
corpus position: `berkeleyaligner_unsupervised/example/train/corpus.{en,fr}` **(this corpus will be uploaded)** <br/> 
put these two lines in the file `align`: <br/>
OPTS="-server -mx10g -ea"<br/>
java $OPTS -jar berkeleyaligner.jar ++$CONF $@ -saveLexicalWeights true <br/>
command: `./berkeleyaligner_unsupervised/align example.conf` <br/>
time used: 6h <br/>
main output: `berkeleyaligner_unsupervised/output/: 1.lexweights, 2.lexweights, training.align` 

generate serialized file: `preparation/berkeley_wordId_lexTable.py` <br/>
output: `pickle_res/{en,fr}_word_id.p, {en,fr}_entropy.p, berkeley_{forward,reverse}_table.p` <br/>
 
4. **ConceptNet Numberbatch embeddings** 

input: [multilingual-numberbatch-17.06.txt](https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz) ([documentation](https://github.com/commonsense/conceptnet-numberbatch)) <br/>
script: `preparation/CNet_embeddings.py` <br/>
output: `pickle_res/CNet_enfr_embeddings.p`

5. **Correspondence between sentence_ID (in corpus) and line_index**

Used to find corresponding parsing information of each sentence. <br/> 
script: `preparation/lineID_dico.py` <br/>
output: `pickle_res/line_parseId_dict.p`  

6. **Syntactic parsing**

English and French dependency and constituency parsing: <br/>
input: `txt_res/{en,fr}_noID.txt`, `corenlp_properties/{en_parsing, fr_parsing}.properties` <br/>
command: `stanford-corenlp-full-2018-10-05/corenlp.sh -props {en_parsing, fr_parsing}.properties` <br/>

We can also use [Bonsai parser](http://alpage.inria.fr/statgram/frdep/fr_stat_dep_parsing.html) for French syntactic parsing, which is much faster. <br/>
command: <br/>
`sh bonsai_v3.2/bin/bonsai_bky_parse_via_clust.sh -f const -n fr_noID.txt > fr_noID_const_bonsai.txt`  <br/>
`sh bonsai_v3.2/bin/bonsai_bky_parse_via_clust.sh -f ldep -n fr_noID.txt > fr_noID_dep_bonsai.txt` 

In this work for the French corpus, we use *Stanford Corenlp* for dependency parsing to share the same set of relation tags with English; and *Bonsai* for constituency parsing because it's faster. However, it seems that *Bonsai* has problem in parsing very long input (we had one entry containing several sentences inside), so we completed its parsing result using *Stanford Corenlp*. 

Write each sentence's syntactic parsing information in a separate file: <br/>
`python txt_res/parsing_oneSentence_perFile.py` <br/>
Separate English and French constituency and dependency information in separate files: <br/>
`python txt_res/extract_cons_dep.py`  <br/>
Transform English and French constituency parsing information, to one word per line format and add word index: <br/>
`python txt_res/transform_cons.py`

7. **ConceptNet assertions**

input: [conceptnet-assertions-5.6.0.csv](https://github.com/commonsense/conceptnet5/wiki/Downloads)  <br/>
This is a multilingual resource, we used regex to extract en-en, en-fr, fr-fr assertions, which you can download [here](https://www.dropbox.com/s/x6sybtkus6gg37o/conceptNet-assertions.zip?dl=0). 

generate serialized files: `preparation/cnet_assertions.py => {en_en, en_fr, fr_fr}_assert_{direct, reverse}.p`  

## Manual alignment information 

`txt_res/alignment.txt`: this file contains manual word alignment and translation process category information 

## Extract features

`cd classification/` <br/>
e.g.: `python get-feature.py ../txt_res/1:1.txt` 

## Train and evaluate classifiers by cross-validation 


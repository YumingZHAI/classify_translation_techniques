## Preparation work

Serialized large volume files (*.p files) are not uploaded here, the procedures used to generate them are described below. (=> or upload all the *.p files in a zipped directory)

1. **PoS tagging** 

tool: [stanford-corenlp-full-2018-10-05](https://stanfordnlp.github.io/CoreNLP/download.html) <br/>
input: `txt_res`/{en,fr}.txt, `corenlp_properties`/{en_pos_lemma, fr_pos}.properties <br/>
command: `stanford-corenlp-full-2018-10-05/corenlp.sh -props {en_pos_lemma, fr_pos}.properties` <br/>
output: `txt_res`/{en,fr}.txt.conll 

2. **Lemmatisation** 

EN: <br/>
tool: [stanford-corenlp-full-2018-10-05](https://stanfordnlp.github.io/CoreNLP/download.html) <br/>
input: `txt_res`/en.txt, `corenlp_properties`/en_pos_lemma.properties <br/>
command: `stanford-corenlp-full-2018-10-05/corenlp.sh -props en_pos_lemma.properties` <br/>
then script: `lemmatisation`/extract-lemma-corenlp.py <br/>
output: `lemmatisation`/ENlemme_from_corenlp.txt <br/>

FR: <br/>
tool: [tree-tagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) (because stanford corenlp doesn't provide French lemmatization yet) <br/>
input: `txt_res`/fr_noID.txt  <br/>
script: `lemmatisation`/fr-lemmatize.py <br/>
output: `lemmatisation`/fr_lemma.txt  <br/>

`lemmatisation`/pickle_lemma.py => `pickle_res`/{eng,fr}_lemma.p <br/>

3. **Berkeley unsupervised word aligner**

input: an English-French parallel corpus composed of TED Talks and a part of Paracrawl corpus (in total 1.8M parallel sentence pairs and 41M English tokens)  <br/>
corpus position: `berkeleyaligner_unsupervised/example/train` <br/> 
command: `berkeleyaligner_unsupervised/align example.conf` <br/>
time used: 6h  <br/>
output: 1.lexweights, 2.lexweights, training.align 

then script: `preparation/berkeley_wordId_lexTable.py` <br/>
output: `pickle_res`/{en,fr}_word_id.p, {en,fr}\_entropy.p, berkeley\_{forward,reverse}_table.p <br/>
 

4. **ConceptNet embeddings** 

input: [multilingual-numberbatch-17.06.txt](https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz) <br/>
script: `preparation/CNet_embeddings.py` <br/>
output: `pickle_res`/CNet_enfr_embeddings.p

## Extract features

`cd classification/` <br/>
e.g.: `python get-feature.py ../txt_res/1:1.txt` 


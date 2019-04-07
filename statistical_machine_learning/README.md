**Corpus preprocessing**

1. pos tagging <br/>

tool: stanford-corenlp-full-2018-10-05 <br/>
input: txt_res/{en,fr}.txt, {en,fr}.properties (to be cleaned) (conll output format) <br/>
command: ./corenlp.sh -props {en,fr}.properties <br/>
output: {en,fr}.txt.conll

2. lemmatisation (cd lemmatisation/)

EN: <br/>
tool: stanford corenlp <br/>
input: en.txt, en.properties (to be cleaned) (conll output format) <br/>
script: extract-lemma-corenlp.py <br/>
output: ENlemme_from_corenlp.txt <br/>

FR: <br/>
tool: tree-tagger (because corenlp doesn't provide French lemmatization now) <br/>
input: fr_noID.txt  <br/>
script: fr-lemmatize.py <br/>
output: fr_lemma.txt  <br/>

pickle_lemma.py => ../pickle_res/{eng,fr}_lemma.p <br/>

3. berkeley unsupervised word aligner 

input: an English-French parallel corpus composed of TED Talks and a part of Paracrawl corpus (in total 1.8M parallel sentence pairs and 41M English tokens)  <br/>
corpus position: berkeleyaligner_unsupervised/example/train <br/> 
command: ./align example.conf <br/>
time used: 6h  <br/>
output: 1.lexweights, 2.lexweights, training.align 

script: preparation/berkeley_wordId_lexTable.py <br/>
output: {en,fr}_word_id.p, {en,fr}\_entropy.p, berkeley\_{forward,reverse}_table.p <br/>
 

**Extract features**

cd classification/ <br/>
e.g.: python get-feature.py ../txt_res/1:1.txt 
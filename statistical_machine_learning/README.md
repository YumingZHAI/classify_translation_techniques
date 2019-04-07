**Corpus preprocessing**

1. pos tagging <br/>

tool: stanford-corenlp-full-2018-10-05 <br/>
input: txt_res/{en,fr}.txt, {en,fr}.properties (to be cleaned) (conll output format) <br/>
command: ./corenlp.sh -props {en,fr}.properties <br/>
output: {en,fr}.txt.conll

2. lemmatization 

EN: <br/>
tool: stanford corenlp <br/>

FR: <br/>
tool: tree-tagger (because corenlp doesn't provide French lemmatization now) <br/>
input: fr_noID.txt  <br/>
script: lemmatisation/fr-lemmatize.py 

pickled files 

**Extract features**

cd classification/ <br/>
e.g.: python get-feature.py ../txt_res/1:1.txt 
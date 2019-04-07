**Corpus preprocessing**

1. pos tagging + lemmatization for EN, pos tagging for FR <br/>
tool: stanford-corenlp-full-2018-10-05 <br/>
input: txt_res/{en,fr}.txt, {en,fr}.properties (conll output format) <br/>
command: ./corenlp.sh -props {en,fr}.properties <br/>
output: {en,fr}.txt.conll

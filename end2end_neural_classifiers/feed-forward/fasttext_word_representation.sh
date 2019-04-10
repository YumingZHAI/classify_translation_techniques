#!/bin/bash

src_text=../dataset/en-fr-corpus/fasttext-normalized/en-3m-normalized
tgt_text=../dataset/en-fr-corpus/fasttext-normalized/fr-3m-normalized
src_model=../dataset/en-fr-corpus/fasttext-normalized/english-normalized
tgt_model=../dataset/en-fr-corpus/fasttext-normalized/french-normalized

../../fastText-0.2.0/fasttext skipgram -input $src_text -output $src_model \
			   -minCount 5 \
			   -ws 5 \
			   -minn 3 \
			   -maxn 6
../../fastText-0.2.0/fasttext skipgram -input $tgt_text -output $tgt_model \
			   -minCount 5 \
			   -ws 5 \
			   -minn 3 \
			   -maxn 6

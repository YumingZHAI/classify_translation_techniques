import pickle
import numpy as np 

CNet_enfr_embeddings = {}
with open('../txt_res/multilingual-numberbatch-17.06.txt') as file:
	for line in file:
		# /c/en/word embedding_vector
		if '/c/en' in line or '/c/fr' in line:
			line = line.strip('\n')
			term = line.split(' ')[0]
			# Convert the input to an array
			coefs = np.asarray(line.split(' ')[1:], dtype='float32')
			CNet_enfr_embeddings[term] = coefs

pickle.dump(CNet_enfr_embeddings, open("../pickle_res/CNet_enfr_embeddings.p", "wb"))

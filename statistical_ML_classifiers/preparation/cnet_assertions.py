# transform ConceptNet's (en-fr, en-en, fr-fr)assertions.csv to direct & reverse dictionary 
# which means direct[start] = [ends], reverse[end] = [starts]
# given a query word, retrieve results which is an union of both direct and reverse dictionary's values 

# original file downloaded: conceptnet-assertions-5.6.0.csv (32755210)  (on the internal server Ponge)
# then use regex to extract
# all-en-fr             7007826 (lines)
# uniq-en-fr assertions 6989530
# en-en assertions      3098816
# fr-fr                 3369426
# en-fr                 539584    (in fact exist cases: /c/fr/English words!)

import re
import pickle

def build_dic(input):   # transform assertions csv file to 2 dics, in two directions
	direct_dic = {}
	reverse_dic = {}
	with open(input, "r") as file:
		for line in file:
			line = line.strip('\n')
			# / a / [ / r / Antonym /, / c / en / adjacent / a /, / c / fr / distant /] / r / Antonym / c / en / adjacent / a / c / fr / distant
			# {"dataset": "/d/wiktionary/fr", "license": "cc:by-sa/4.0",
			#  "sources": [{"contributor": "/s/resource/wiktionary/fr", "process": "/s/process/wikiparsec/1"}],
			#  "weight": 1.0}
			m = re.match(r'(/.*?]).*?{.*?weight": (.*?)}$', line)
			triple = m.group(1)
			weight = m.group(2)
			# /a/[/r/Antonym/,/c/en/adjacent/a/,/c/fr/distant/]
			relation = triple.split(',')[0].lstrip('/a/[/r/').rstrip('/')
			# before: pb using ',' as delimiter, because some segment has ',' in it
			n = re.search(r'/c/(en|fr)/(.*?)/([n,v,a,s,r]/)?,/c/(en|fr)/(.*?)/([n,v,a,s,r]/)?]', line)
			start = n.group(2)
			end = n.group(5)
			# print(start, end, relation, weight)

			if direct_dic.get(start) == None:
				end_value = []
				end_value.append(end + "#" + relation + "#" + weight)
				direct_dic[start] = end_value
			else:
				direct_dic[start].append(end + "#" + relation + "#" + weight)

			if reverse_dic.get(end) == None: 
				start_value = []
				start_value.append(start + "#" + relation + "#" + weight)
				reverse_dic[end] = start_value
			else:
				reverse_dic[end].append(start + "#" + relation + "#" + weight)

	return direct_dic, reverse_dic


en_fr_assert_direct, en_fr_assert_reverse = build_dic("../txt_res/en-fr-assertions.csv")
pickle.dump(en_fr_assert_direct, open("../pickle_res/en_fr_assert_direct.p", "wb"))
pickle.dump(en_fr_assert_reverse, open("../pickle_res/en_fr_assert_reverse.p", "wb"))

en_en_assert_direct, en_en_assert_reverse = build_dic("../txt_res/en-en-assertions.csv")
pickle.dump(en_en_assert_direct, open("../pickle_res/en_en_assert_direct.p", "wb"))
pickle.dump(en_en_assert_reverse, open("../pickle_res/en_en_assert_reverse.p", "wb"))

fr_fr_assert_direct, fr_fr_assert_reverse = build_dic("../txt_res/fr-fr-assertions.csv")
pickle.dump(fr_fr_assert_direct, open("../pickle_res/fr_fr_assert_direct.p", "wb"))
pickle.dump(fr_fr_assert_reverse, open("../pickle_res/fr_fr_assert_reverse.p", "wb"))




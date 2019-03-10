"""
author: Yuming ZHAI
This script calculates features for an input file with each entry under the form:
causes # fait que : 1556_29_32, 33 : generalization
                    sentenceID_english word indices_french word indices
"""
import os, sys, re
import pprint
import Levenshtein
from scipy import spatial
from datetime import datetime
# classification and utils are in the same level, add this sys.path to import module
sys.path.append('../')
# use all my functions, load pickled files
from utils.my_functions import *

print(str(datetime.now()) + " load pickle files")

print(str(datetime.now()) + " load pos files")   # by Stanford CoreNLP
en_dico_pos = dico_feature("../txt_res/en.txt.conll")
fr_dico_pos = dico_feature("../txt_res/fr.txt.conll")

source = sys.argv[1]

# encode label to int
if "4class" in source:   # without equiv
    label_list = ['generalization', 'particularization', 'modulation', 'contain_transposition']
elif "5class" in source:
    label_list = ['equivalence', 'generalization', 'particularization', 'modulation', 'contain_transposition']
elif "6class" in source:
    label_list = ['literal', 'equivalence', 'generalization', 'particularization', 'modulation', 'contain_transposition']
elif "1:1" in source or "2:1" in source or "3:1" in source:
    label_list = ['literal', 'non_literal']
elif "EL" in source:   # to be able to use "LET in source" below
    label_list = ['literal+equi', 'non_LE']
elif "LET" in source:    # group L, E, transposition
    label_list = ['literal+equi+transp', 'non_LET']

label_id = {}
i = 0
for label in label_list:
    label_id[label] = i
    i += 1

pos_count = open("../features/pos_count.txt", 'w')
pos_cosinus = open("../features/pos_cosinus.txt", 'w')
pos_cos_content = open("../features/pos_cos_content.txt", 'w')
entropy = open("../features/entropy.txt", 'w')
lexWeighting = open("../features/lexWeighting.txt", 'w')
surface = open("../features/surface.txt", 'w')
CNetEmbed = open("../features/CNetEmbed.txt", 'w')
delta_literal = open("../features/delta_literal.txt", 'w')
literal_unaligned = open("../features/literal_unaligned.txt", "w")
posChange = open("../features/posChange.txt", "w")
CNetLink = open("../features/CNetLink.txt", "w")
percentDeriv = open("../features/percentDeriv.txt", 'w')
constituency = open("../features/constituency.txt", 'w')
internal_dep = open("../features/internal_dep.txt", 'w')
external_dep = open("../features/external_dep.txt", 'w')
label = open("../features/label.txt", 'w')

with open("../txt_res/alignment.txt") as align:
    aligns = align.readlines()

abc = 0
# open the file of examples: e.g. "../txt_res/balanced.txt"
with open(os.path.abspath(source)) as file:
    for line in file:
        # ex. all the people in the world # la population mondiale : 748_3, 4, 5, 6, 7, 8_3, 4, 5 : equivalence
        if abc == 0: print(str(datetime.now()) + " FOR EACH LINE OF TRAINING EXAMPLE:")

        if abc == 0: print(str(datetime.now()) + " prepare segment pair information")
        # ---------------------------------------------------------------------------- prepare segment information
        m = re.match(r'(.*?\#.*?) : (\d+)_(.*?)_(.*?) : (.*)$', line)
        text = m.group(1)
        lineID = m.group(2)
        # word id
        engID = m.group(3)
        foreignID = m.group(4)
        category = m.group(5)

        english = text.split(" # ")[0]
        foreign = text.split(" # ")[1]

        # original length with all words
        en_len = len(english.split(' '))
        for_len = len(foreign.split(' '))

        # ---------------------------------------------------------------- get the original PoS tag sequence
                # then MAP language-specific PoS tags to universal ones
        tabEn = engID.split(", ")   # attention there's a space after the comma

        # key: lineID, value: dico[wordID]=PoSTag
        sentPosDico = en_dico_pos[lineID]

        eng_pos = ""
        for indice in tabEn:
            word_pos = sentPosDico[int(indice)]
            # map English pos tag set to universal PoS tag set
            eng_pos += map_pos_tag(word_pos, 'en') + " "

        # remove whitespace at right end
        eng_pos = eng_pos.rstrip()
        # -----------------

        tabFr = foreignID.split(", ")
        sentPosDico = fr_dico_pos[lineID]

        fr_pos = ""
        for indice in tabFr:
            # by using Stanford CoreNLP, French PoS tagging uses already the universal PoS tag set
            fr_pos += sentPosDico[int(indice)] + " "

        fr_pos = re.sub(r'\bCONJ\b', 'CCONJ', fr_pos)   # only this one is not coherent with the universal pos tag set
        fr_pos = fr_pos.rstrip()

        # ex.DET DET NOUN ADP DET NOUN
        # print(eng_pos)
        # ex.DET NOUN ADJ
        # print(foreign, '#', fr_pos)

        # ------------------------------------------------------------ keep POS tag information with words

        eng_info = []
        for x, y, z in (zip(english.split(' '), eng_pos.split(' '), engID.split(', '))):
            eng_info.append(x + "/" + y + "/" + z)
        # ex. ['all/DET/3', 'the/DET/4', 'people/NOUN/5', 'in/ADP/6', 'the/DET/7', 'world/NOUN/8']

        fr_info = []
        for x, y, z in (zip(foreign.split(' '), fr_pos.split(' '), foreignID.split(', '))):
            fr_info.append(x + "/" + y + "/" + z)
        # ex. ['la/DET/3', 'population/NOUN/4', 'mondiale/ADJ/5']

        # print(eng_info)
        # print(fr_info)

        # # -------------------------------------------------- get only content words and their indices (both are tables)
        #
        en_content, en_content_indice = get_content_words(eng_info, eng_pos)
        fr_content, fr_content_indice = get_content_words(fr_info, fr_pos)

        # ex. ['people', 'world'] ['5', '8']
        # print(en_content, en_content_indice)

        # ex. ['population', 'mondiale'] ['4', '5']
        # print(fr_content, fr_content_indice)
        #
        # # -------------------------------------------------------------------- get lemmatised all words (as string)
        #
        eng_lemma_tab = eng_lemma[int(lineID)]    # eng_lemma[lineID] = table of lemma
        fr_lemma_tab = fr_lemma[int(lineID)]

        eng_lemmatised = ''
        for i in engID.split(', '):
            eng_lemmatised += eng_lemma_tab[int(i)] + ' '

        fr_lemmatised = ''
        for i in foreignID.split(', '):
            fr_lemmatised += fr_lemma_tab[int(i)] + ' '

        eng_lemmatised = eng_lemmatised.rstrip(' ')
        fr_lemmatised = fr_lemmatised.rstrip(' ')

        # print(eng_lemmatised)
        # ex. le population mondial
        # print(fr_lemmatised)

        # # ------------------------------------------------------------------ get lemmatised content words (as string)

        # because we have content words' id
        eng_content_lemma = ''
        for i in en_content_indice:
            eng_content_lemma += eng_lemma_tab[int(i)] + ' '

        fr_content_lemma = ''
        for i in fr_content_indice:
            fr_content_lemma += fr_lemma_tab[int(i)] + ' '

        eng_content_lemma = eng_content_lemma.rstrip(' ')
        fr_content_lemma = fr_content_lemma.rstrip(' ')

        # ex. population mondial
        # print(fr_content_lemma)

        ############################################################# above is preparation work

        if abc == 0: print(str(datetime.now()) + " feature comparing POS tag number of occurrences")
        # # ----------------------------------------------------------------------- universal POS tag N-hot encoding
        # turn pos string to their ID representation (a list of integer)
        en_id_pos = univ_pos_id(eng_pos)
        fr_id_pos = univ_pos_id(fr_pos)

        dict_en = dict((x, en_id_pos.count(x)) for x in set(en_id_pos))
        dict_fr = dict((x, fr_id_pos.count(x)) for x in set(fr_id_pos))

        # ex. {1: 1, 5: 3, 7: 2}
        # print(dict_en)

        en_encoding = [0] * 17  # in total 17 universal pos tags
        fr_encoding = [0] * 17
        # fill N-hot encoding vector: count presence of each POS tag
        for key in dict_en.keys():
            en_encoding[key] = dict_en[key]

        for key in dict_fr.keys():
            fr_encoding[key] = dict_fr[key]

        for x in en_encoding:
            print(x, file=pos_count, end="\t")
        for y in fr_encoding:
            print(y, file=pos_count, end="\t")

        # -------------------------------------------------- N-hot encoding POS tag feature, cosine similarity
        if abc == 0: print(str(datetime.now()) + " feature comparing POS tag vector cosine similarity")

        cosine = 1 - spatial.distance.cosine(en_encoding, fr_encoding)
        print(cosine, file=pos_cosinus, end="\t")
        #
        # # ------------------------------------------------ this pos tag cosine similarity is only on content words
        #
        content_en_encoding = en_encoding
        content_fr_encoding = fr_encoding

        # these are considered as no-content words' universal POS tag id
        indices = [1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16]
        for i in indices:
            content_en_encoding[i] = 0
            content_fr_encoding[i] = 0

        #  if all are non-content words, use its original encoding
        if all(item == 0 for item in content_en_encoding) or all(item == 0 for item in content_fr_encoding):
            # the original cosine with all pos tags (no filter)
            print(cosine, file=pos_cos_content, end="\t")
        else:
            content_cosine = 1 - spatial.distance.cosine(content_en_encoding, content_fr_encoding)
            print(content_cosine, file=pos_cos_content, end="\t")

        # # -------------------------------------------------------------------------------- simple surface features
        if abc == 0 : print(str(datetime.now()) + " feature on surface (nb tokens, Levenshtein)")
        #
        # number of all tokens
        print(en_len, file=surface, end="\t")
        print(for_len, file=surface, end="\t")
        ratio1 = 1.0 * en_len / for_len
        print(ratio1, file=surface, end="\t")
        ratio2 = 1.0 * for_len / en_len
        print(ratio2, file=surface, end="\t")

        # range of different values instead of an artificial threshold
        leven_distance = Levenshtein.distance(english, foreign)
        print(leven_distance, file=surface, end="\t")

        # # -------------------------------------------------------------------------- lexical weighting between content words
        # # assume the alignment between segments is n-m, bloc to bloc. because our manual alignment for MWE is like this
        #
        # use berkeley's lexical translation probability value, give minimum value 0.0000001 as minimum value
        if abc == 0: print(str(datetime.now()) + " feature on bidirectional lexical weighting")

        dir_content_lw = get_direct_lexical_weighting(fr_content, en_content, en_word_id, fr_word_id, ber_dir_table)
        rev_content_lw = get_reverse_lexical_weighting(fr_content, en_content, en_word_id, fr_word_id, ber_rev_table)

        print(dir_content_lw, file=lexWeighting, end="\t")
        print(rev_content_lw, file=lexWeighting, end="\t")

        dir_content_lemma_lw = get_direct_lexical_weighting(fr_content_lemma.split(' '), eng_content_lemma.split(' '), en_word_id, fr_word_id, ber_dir_table)
        rev_content_lemma_lw = get_reverse_lexical_weighting(fr_content_lemma.split(' '), eng_content_lemma.split(' '), en_word_id, fr_word_id, ber_rev_table)

        print(dir_content_lemma_lw, file=lexWeighting, end="\t")
        print(rev_content_lemma_lw, file=lexWeighting, end="\t")

        # ----------------------------------------------------  word translation entropy from berkeley's lexical translation table
        # before I used word frequency from EN and FR comparable subtitle corpus
        # they are independent with regard to translation so it can mislead the algorithm
        #
        if abc == 0: print(str(datetime.now()) + " feature on average word translation entropy")

        en_m_entropy = average_entropy(en_content, en_word_id, en_entropy)
        fr_m_entropy = average_entropy(fr_content, fr_word_id, fr_entropy)
        print(en_m_entropy, file=entropy, end="\t")
        print(fr_m_entropy, file=entropy, end="\t")

        en_lem_m_entropy = average_entropy(eng_content_lemma.split(' '), en_word_id, en_entropy)
        fr_lem_m_entropy = average_entropy(fr_content_lemma.split(' '), fr_word_id, fr_entropy)
        print(en_lem_m_entropy, file=entropy, end="\t")
        print(fr_lem_m_entropy, file=entropy, end="\t")

        # # --------------------------------------------------------------------------- conceptNet embedding features
        # don't use API, use local files to consult assertions and embeddings
        # text_to_uri.py (improvements done), wordfreq-2.0/ should be in the same directory
        if abc == 0: print(str(datetime.now()) + " feature on ConceptNet embeddings")

        # get embedding for the entire segment (MWE), otherwise average embeddings on only content words
        cnet_embedding(english, foreign, en_content, fr_content, CNetEmbed)
        cnet_embedding(eng_lemmatised, fr_lemmatised, eng_content_lemma.split(' '), fr_content_lemma.split(' '), CNetEmbed)

        # ---------------------------------------------------------------------------- constituency parsing features
        if abc == 0: print(str(datetime.now()) + " feature on constituency parsing")

        fileID = line_parseId_dict[lineID]
        # print(fileID)

        newTabEn = [int(x) + 1 for x in tabEn]
        newTabFr = [int(x) + 1 for x in tabFr]

        enCons = "../txt_res/en_cons_transform/" + str(fileID) + ".txt"
        frCons = "../txt_res/fr_cons_transform/" + str(fileID) + ".txt"

        cons_en, type_en = en_constituent_parsing(enCons, newTabEn)
        cons_fr, type_fr = fr_constituent_parsing(frCons, newTabFr)

        # print(cons_en)
        # print(cons_fr)

        if type_en == type_fr == "pos":
            if cons_en == cons_fr:
                print(0, file=constituency, end="\t")
                # print("same pos") # 0
            else:
                print(1, file=constituency, end="\t")
                # print("diff pos") # 1
        elif type_en ==  type_fr == "const":
            if cons_en == cons_fr:
                print(0, file=constituency, end="\t")
                # print("same const") # 0
            else:
                print(1, file=constituency, end="\t")
                # print("diff const") # 1
        elif type_en == "pos" and type_fr == "const":
            if map_pos_const[cons_en] == cons_fr:
                print(0, file=constituency, end="\t")
                # print("same category") # 0
            else:
                print(1, file=constituency, end="\t")
                # print("diff category") # 1
        elif type_en == "const" and type_fr == "pos":
            if map_pos_const[cons_fr] == cons_en:
                print(0, file=constituency, end="\t")
                # print("same category") # 0
            else:
                print(1, file=constituency, end="\t")
                # print("diff category") # 1
        else:
            print("problem on constituency parsing! " + line + ", fileID: " + str(fileID))

        # ---------------------------------------------------------------------------- internal dependency parsing features

        if abc == 0: print(str(datetime.now()) + " feature on internal dependency parsing")

        enDep = "../txt_res/en_dep_parsing/" + str(fileID) + ".txt"
        frDep = "../txt_res/fr_dep_parsing/" + str(fileID) + ".txt"

        en_int_rel_vector = en_internal_dep(enDep, newTabEn)
        fr_int_rel_vector = fr_internal_dep(frDep, newTabFr)

        for x in en_int_rel_vector:
            print(x, file=internal_dep, end="\t")

        for x in fr_int_rel_vector:
            print(x, file=internal_dep, end="\t")

        cosine = 1 - spatial.distance.cosine(en_int_rel_vector, fr_int_rel_vector)
        print(cosine, file=internal_dep, end="\t")

        # ---------------------------------------------------------------------------- external dependency parsing features

        if abc == 0: print(str(datetime.now()) + " feature on external dependency parsing")

        en_linked, en_dic_link = en_external_dep(enDep, newTabEn)
        fr_linked, fr_dic_link = fr_external_dep(frDep, newTabFr)
        en_rel_vector, fr_rel_vector = compare_external_dep(en_linked, fr_linked, en_dic_link, fr_dic_link, aligns, fileID)

        for x in en_rel_vector:
            print(x, file=external_dep, end="\t")

        for x in fr_rel_vector:
            print(x, file=external_dep, end="\t")

        # don't put this cosine feature, because it brings many "nan" values, reason:
        # either the two segments don't have any external dependency
        # either after filtering the aligned words, there remain no dependencies to be compared
        # cosine = 1 - spatial.distance.cosine(en_rel_vector, fr_rel_vector)
        # print(cosine, file=external_dep, end="\t")

        # ---------------------------------------------------------------  delta between actual lexical translation vs the most literal translation
        if abc == 0: print(str(datetime.now()) + " feature on delta from the most literal translation & literal_unaligned ratio")
        # use lemma isn't a good idea

        direct_delta(english.split(' '), foreign.split(' '), en_word_id, fr_word_id, ber_dir_table, delta_literal, literal_unaligned)
        reverse_delta(english.split(' '), foreign.split(' '), en_word_id, fr_word_id, ber_rev_table, delta_literal, literal_unaligned)

        # # ---------------------------------------------------------------------- feature of PoS changing pattern

        if abc == 0: print(str(datetime.now()) + " feature of PoS changing pattern")

        matchPattern = 0
        if check_pos_pattern(eng_pos, fr_pos) == "1":
            matchPattern = 1

        # print pos and sort to check new patterns!
        # print(eng_pos, '_', english, '\t', fr_pos, '_', foreign, ' # ', matchPattern)

        print(matchPattern, file=posChange, end="\t")

        # ---------------------------------------------------------------------- feature for (in)direct link in ConceptNet

        if abc == 0: print(str(datetime.now()) + " feature for (in)direct link in ConceptNet")

        en_blacklist = ['be', 'make', 'get', 'the', 'a', 'an', 'of', 'in', 'to', 'with', 'than', 'at', '', 's', 'it']
        fr_blacklist = ['être', 'faire', 'avoir', 'le', 'la', 'les', 'de', 'des', 'd', 'du', 'que', 'qui', 'un', 'se',
                        'à', 'au', 'avec', 'pour', 'par', 'y', 'dans', 'je', 'comme', 'en', 'et']

        e = re.sub(r" |'|-", "_", english.lower())
        f = re.sub(r" |'|-", "_", foreign.lower())
        f = re.sub(r"_du$", "_de", f)
        # print("\toriginal pair: ", e, f)

        c = re.sub(r" |'|-", "_", eng_lemmatised.lower())
        d = re.sub(r" |'|-", "_", fr_lemmatised.lower())
        if 'd\'' in foreign and 'de_' in d:
            d = re.sub(r"de_", "d_", d)
        # print("\tlemmatised pair: ", c, d)

        # a, b: lemmatised segment with black words (from the hand made list) filtered
        a = [token for token in c.split('_') if token not in en_blacklist]
        b = [token for token in d.split('_') if token not in fr_blacklist]
        # if no lemmatised word remains after the filtering, take the original c and d
        if not a:
            a = c.split('_')
        if not b:
            b = d.split('_')
        # print("\tfiltered lemmatised pair: ", a, b)

        # entire segment link, under three different forms
        # result: 0 directly linked, 1 indirectly linked via French bridge word
        # 2 : not linked at all

        # print(e, f)
        # print(c, d)
        # print(a, b)
        result_ef = entire_seg_link(e, f)
        if result_ef == 0 or result_ef == 1:   # directly or indirectly linked
            # print(result_ef, "original")
            print(result_ef, file=CNetLink, end="\t")
        else:
            result_cd = entire_seg_link(c, d)
            if result_cd == 0 or result_cd == 1:
                # print(result_cd, "lemmatised")
                print(result_cd, file=CNetLink, end="\t")
            else:
                result_entire_ab = entire_seg_link('_'.join(a), '_'.join(b))
                if result_entire_ab == 0 or result_entire_ab == 1 :
                    # print(result_entire_ab, ", filtered lemmatised")
                    print(result_entire_ab, file=CNetLink, end="\t")
                else:
                    # print(2)
                    print(2, file=CNetLink, end="\t")
                    # not linked (under three forms, using entire_seg_link: not directly or indirectly linked in CNet)
                    # can also because the english segment isn't present in the resource

        # ---------------------------------------------------------------------- feature for derivation percent from ConceptNet

        if abc == 0: print(str(datetime.now()) + " feature for derivation percent from ConceptNet")

        ## compute percentage of indirectly linked words (via French bridge words) on filtered lemmatised words
        # percent_ef = link_percent(e.split(' '), f.split(' '))
        # percent_cd = link_percent(c.split(' '), d.split(' '))
        # print(percent_ef)
        # print(percent_cd)
        percent_ab = link_percent(a, b)
        # print("word derivation percent: ", percent_ab)
        print(percent_ab, file=percentDeriv, end="\t")

        # # -------------------------------------------------------------------------  write labels
        if abc == 0: print(str(datetime.now()) + " write label info")

        if "4class" in source or "5class" in source or "6class" in source :
            if category == 'transposition':
                category = 'contain_transposition'
            elif category == 'modulation_transposition':
                category = 'contain_transposition'
        elif "1:1" in source or "2:1" in source or "3:1" in source:
            if category != 'literal':
                category = 'non_literal'
        elif "EL" in source:
            if category == 'literal':
                category = 'literal+equi'
            elif category == 'equivalence':
                category = 'literal+equi'
            else:
                category = 'non_LE'
        elif "LET" in source:
            if category == 'literal':
                category = 'literal+equi+transp'
            elif category == 'equivalence':
                category = 'literal+equi+transp'
            elif category == 'transposition':
                category = 'literal+equi+transp'
            else:
                category = 'non_LET'

        category_id = label_id[category]   # turn category from string to integer
        print(category_id, file=label, end="")

        # # -------------------------------------------------------------------------

        pos_count.write('\n')
        pos_cosinus.write('\n')
        pos_cos_content.write('\n')
        entropy.write('\n')
        lexWeighting.write('\n')
        surface.write('\n')
        CNetEmbed.write('\n')
        delta_literal.write('\n')
        literal_unaligned.write('\n')
        posChange.write('\n')
        CNetLink.write('\n')
        percentDeriv.write('\n')
        constituency.write('\n')
        internal_dep.write('\n')
        external_dep.write('\n')
        label.write('\n')

        abc += 1

pos_count.close()
pos_cosinus.close()
pos_cos_content.close()
entropy.close()
lexWeighting.close()
surface.close()
CNetEmbed.close()
delta_literal.close()
literal_unaligned.close()
posChange.close()
CNetLink.close()
percentDeriv.close()
constituency.close()
internal_dep.close()
external_dep.close()
label.close()



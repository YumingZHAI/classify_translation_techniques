"""
author: Yuming ZHAI
This script groups different functions used to calculate features.
"""

import re
import pickle
import Levenshtein
from scipy import spatial
import numpy as np
import operator

import utils.text_to_uri as t2u
from utils.linguistic_variables import *

# load serialized line parse ID dico
line_parseId_dict = pickle.load(open("../pickle_res/line_parseId_dict.p", "rb"))

# load serialized ConceptNet embeddings file
CNet_enfr_embeddings = pickle.load(open("../pickle_res/CNet_enfr_embeddings.p", "rb"))

# load serialized ConceptNet assertions file
en_fr_assert_direct = pickle.load(open("../pickle_res/en_fr_assert_direct.p", "rb"))
en_fr_assert_reverse = pickle.load(open("../pickle_res/en_fr_assert_reverse.p", "rb"))

en_en_assert_direct = pickle.load(open("../pickle_res/en_en_assert_direct.p", "rb"))
en_en_assert_reverse = pickle.load(open("../pickle_res/en_en_assert_reverse.p", "rb"))

fr_fr_assert_direct = pickle.load(open("../pickle_res/fr_fr_assert_direct.p", "rb"))
fr_fr_assert_reverse = pickle.load(open("../pickle_res/fr_fr_assert_reverse.p", "rb"))

# load serialized eng, french lemmatised annotated corpus
eng_lemma = pickle.load(open("../pickle_res/eng_lemma.p", "rb"))
fr_lemma = pickle.load(open("../pickle_res/fr_lemma.p", "rb"))

# load serialized word id (from berkeley lex weights file)
# if have encoding error when loading, try regenerate pickle file
en_word_id = pickle.load(open("../pickle_res/en_word_id.p", "rb"))
fr_word_id = pickle.load(open("../pickle_res/fr_word_id.p", "rb"))

# load serialized berkeley lexical translation table
ber_dir_table = pickle.load(open("../pickle_res/berkeley_forward_table.p", "rb"))
ber_rev_table = pickle.load(open("../pickle_res/berkeley_reverse_table.p", "rb"))

# load serialized translation entropy, from berkeley word translation table
en_entropy = pickle.load(open("../pickle_res/en_entropy.p", "rb"))
fr_entropy = pickle.load(open("../pickle_res/fr_entropy.p", "rb"))

#------------------------------------------------------------------------------------------------------------

# read pos files
def dico_feature(file_pos):
    dico1 = {}
    with open(file_pos) as file:
        for line in file:
            lineID = line.split(" ")[0].split("/")[0]
            dico2 = {}
            # strip(" \n") for fr.txt.conll ??
            # tab = line.strip(" ?\n").split(" ")[1:]
            tab = line.strip("\n").split(" ")[1:]
            # word id i
            i = 0
            for item in tab:
                # which word id corresponds to which pos
                dico2[i] = item.split("/")[1]
                i += 1
            # key: lineID, value: dico[wordID]=posTag
            dico1[lineID] = dico2
        return dico1


# b: before, a: after
def map_pos_tag(b, language):
    if language == 'en':
        a = en_univ_posTag[b]
    elif language == 'fr':
        a = fr_univ_posTag[b]
    elif language == 'zh':
        a = zh_univ_posTag[b]
    return a

def univ_pos_id(string_pos):
    id_pos = []
    list = string_pos.split(' ')
    for tag in list:
        id = id_univ_postag[tag]
        id_pos.append(id)
    return id_pos      # e.g. AUX VERB ADJ ADP -> [3, 15, 0, 1]

def map_constituent_tag(b, language):
    if language == 'en':
        a = en_const_Tag[b]
    elif language == 'fr':
        a = fr_const_Tag[b]
    # elif language == 'zh':
    #     a = zh_univ_posTag[b]
    return a

def map_dep_tag(b):
    if b in en_fr_dep_rel:
        return(en_fr_dep_rel[b])
    elif re.match(r'^acl:.*', b):
        return ('acl')
    elif re.match(r'^nmod:.*', b):
        return('nmod')
    elif re.match(r'^conj:.*', b):
        return('conj')
    elif re.match(r'^advcl:.*', b):
        return('advcl')
    elif re.match(r'^compound:.*', b):
        return('compound')
    elif re.match(r'^det:.*', b):
        return('det')
    elif re.match(r'^nsubj:.*', b):
        return('nsubj')
    elif re.match(r'^nsubjpass:.*', b):
        return('nsubjpass')
    elif re.match(r'^cc:.*', b):
        return('cc')

#------------------------------------------------------------------------------------------------------------

# ex. ['la/DET/3', 'population/NOUN/4', 'mondiale/ADJ/5'], DET NOUN ADJ
def get_content_words(list_info, list_pos):
    word_tab = []
    indice_tab = []
    for item in list_info:
        word = item.split('/')[0]
        pos_tag = item.split('/')[1]
        indice = item.split('/')[2]
        content = False
        # check whether content pos tags are in list_pos
        for tag in list_pos.split(' '):
            if tag in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
                content = True
        # if there is content word in this segment, only take them
        if content:
            if pos_tag in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
                word_tab.append(word)
                indice_tab.append(indice)
        # if there's no content word, fall back to the original segment !
        else:
            word_tab.append(word)
            indice_tab.append(indice)

    return word_tab, indice_tab


#------------------------------------------------------------------------------------------------------------

# direct: english translated into french
# for each french word, search the proba P(fr|each en)
def get_direct_lexical_weighting(fr_list, en_list, en_word_id, fr_word_id, forward_table):
    # all words are lowercased in lexweights file
    # print(fr_content)
    # print(en_content)
    P_tgt_G_src = 1.0

    for F_token in fr_list:
        lex_score = 0.0
        fr_id = fr_word_id.get(F_token.lower(), None)
        # print("\n")
        for E_token in en_list:
            en_id = en_word_id.get(E_token.lower(), None)
            # word_ids are generated from berkeley word alignment weights file
            if en_id is not None and fr_id is not None:
                a = forward_table.get(en_id)
                b = a.get(fr_id, None)
                if b is not None:
                    lex_score += float(b)
                    # print(F_token, E_token, b)
                else:
                    lex_score += 0.0000001
                    # print("F isn't a translation of E: ", F_token, " ", E_token, " put minimum value")
            else:
                lex_score += 0.0000001
                # print("E, F don't exist ", F_token, " ", E_token, " put minimum value")

        # print("sum P(this foreign|each english): ", lex_score)
        average = lex_score / len(en_list)
        # print("average P(for|en): ", average)
        P_tgt_G_src *= average
    # print("\ndirect lexical weighting: ", P_tgt_G_src)
    return P_tgt_G_src

def get_reverse_lexical_weighting(fr_list, en_list, en_word_id, fr_word_id, reverse_table):
    P_src_G_tgt = 1.0

    for E_token in en_list:
        lex_score = 0.0
        en_id = en_word_id.get(E_token.lower(), None)
        # print("\n")
        for F_token in fr_list:
            fr_id = fr_word_id.get(F_token.lower(), None)
            if en_id is not None and fr_id is not None:
                a = reverse_table.get(fr_id)
                b = a.get(en_id, None)
                if b is not None:
                    lex_score += float(b)
                    # print(E_token, F_token, b)
                else:
                    lex_score += 0.0000001
                    # print("E isn't a translation of F: ", E_token, " ", F_token, " put minimum value")
            else:
                lex_score += 0.0000001
                # print("E, F don't exist ", E_token, " ", F_token, " put minimum value")

        # print("sum P(this English|each foreign): ", lex_score)
        average = lex_score / len(fr_list)
        # print("average P(en|for): ", average)
        P_src_G_tgt *= average
    # print("\nreverse lexical weighting: ", P_src_G_tgt)
    return P_src_G_tgt

#------------------------------------------------------------------------------------------------------------

# respectively compute for English and foreign segment
# segment is a table here
def average_entropy(segment, word_id_dict, entropy_dict):  
    a = 0.0
    i = 0
    entropy = 0.0
    for token in segment:
        token_id = word_id_dict.get(token.lower(), None)
        if token_id is not None:
            entropy = entropy_dict[token_id]
        else:
            # => e.g. after correcting manually misspelled words: traîtrise
            # print("no token id: ", token)
            i += 1
        a += float(entropy)
    # if a token doesn't have id, don't take it into consideration
    b = len(segment)-i
    # print("b: ", b)
    if b == 0:       # overall entropy should also be 0, when none of the words has word id in lexweights table
        b = len(segment)
    # print("all entropy: ", a)
    average = a / b
    # print("average entropy: ", average)
    return average

#------------------------------------------------------------------------------------------------------------

# input eng, fr are table of words
def direct_delta(eng, fr, en_word_id, fr_word_id, forward_table, fdelta, fUnalign):
    # lowercase eng and foreign token to get its id. because they are all lowercased in lexweights file
    most_literal = 0.0
    real_trans = 0.0

    all_max_fr_id = []
    all_fr_id = []
    # unaligned english words: no id (according to berkeley word translation table)
    j = 0
    # unaligned french words: no id (according to berkeley word translation table)
    m = 0
    # unaligned english words: no foreign word in this pair appears in lexical table as its possible translations
    k = 0

    # get each french token's word id (if it's in dictionary)
    for F_token in fr:
        fr_id = fr_word_id.get(F_token.lower(), None)
        if fr_id is not None:
            all_fr_id.append(fr_id)
        else:
            m += 1

    for E_token in eng:
        en_id = en_word_id.get(E_token.lower(), None)
        if en_id is not None:
            # print("\ncurrent english word: " + E_token.lower())
            # trans is a dictionary
            trans = forward_table.get(en_id)
            # sort dictionary by value, reverse order (should import operator)
            sorted_trans = sorted(trans.items(), key=operator.itemgetter(1), reverse=True)
            most_lite_tuple = sorted_trans[0]
            # indice, proba
            indice = most_lite_tuple[0]
            top_proba = most_lite_tuple[1]

            # word = list(fr_word_id.keys())[list(fr_word_id.values()).index(indice)]
            # print("the most literal translation: ", word, top_proba)
            # above: find statistically the most literal translation of an eng word and its corresponding proba

            group_indice = []
            group_proba = []

            # consider the real current translation:
            for F_token in fr:
                fr_id = fr_word_id.get(F_token.lower(), None)
                if fr_id is not None:
                    # if this French word is a possible translation of the current English word
                    # (according to the lexical translation table)
                    if fr_id in trans.keys():
                        fr_proba = trans[fr_id]
                        if float(fr_proba) != 0.0:
                            # print("found translation: " + F_token.lower() + " " + fr_proba)
                            group_indice.append(fr_id)
                            group_proba.append(fr_proba)
                    elif fr_id not in trans.keys():
                        # will be counted later
                        # print("\t\t" + F_token + ": is not amongst the possible translations in lex table")
                        pass
                else:
                    # already counted before by 'm'
                    pass

            # print("Aligned french indice in lexical table: ", group_indice)  # match fr word_id
            # print("Aligned french probas in lexical table: ", group_proba)   # their proba

            # as long as en_id is not None, we can get its most literal translation
            # then we should increment most_literal
            most_literal += float(top_proba)

            # possible translations for this current English word: take the most probable one
            if group_proba:
                max_p = max(group_proba)
                a = group_proba.index(max_p)
                max_word_id = group_indice[a]
                # which french word gets the max proba for current english word
                # it's surely not perfect for n-m alignment: potato -> pomme de terre
                all_max_fr_id.append(max_word_id)

                # comparison of two proba:
                real_trans += float(max_p)
            else:
                # no foreign word in this pair appears in lexical table as current Eng word's possible translations
                k += 1
        else:
            # no English id (according to Berkeley's word translation table)
            j += 1
            # print(eng)

    # print("\nmost_literal ", most_literal)
    # print("real_trans ", real_trans)

    # comparison of most_literal and most_probable
    delta = most_literal - real_trans
    if most_literal == real_trans == 0.0 :
        print(np.NaN, file=fdelta, end="\t")
        # print("nan", eng, fr)
        # print("nan")
    else:
        print(delta, file=fdelta, end="\t")
        # print("Delta(sum) with most literal: ", delta)

    # unaligned french words: here they're not the most probable candidate translation
    diff = [x for x in all_fr_id if x not in all_max_fr_id]
    # penalize unaligned words (english and french)
    unaligned_en = k + j
    unaligned_fr = len(diff) + m
    # eng, fr are lists
    print(unaligned_en/len(eng)*1.0, file=fUnalign, end="\t")
    print(unaligned_fr/len(fr)*1.0, file=fUnalign, end="\t")

    # print("EN words without id (so without most literal translation) ", j)
    # print("EN words without most probable translation ", k)    # when group_proba is empty
    # print("FR words without id ", m)
    # print("FR words which are not the most literal translation ", len(diff))
    # ex. alternatives -> solutions de remplacement     (de remplacement -> they didn't get the highest proba for alternatives)
    # print("------\n")


def reverse_delta(eng, fr, en_word_id, fr_word_id, reverse_table, fdelta, fUnalign):

    most_literal = 0.0
    real_trans = 0.0

    all_max_en_id = []
    all_en_id = []
    j = 0
    m = 0
    k = 0

    # get each english token's word id (if it's in dictionary)
    for E_token in eng:
        en_id = en_word_id.get(E_token.lower(), None)
        if en_id is not None:
            all_en_id.append(en_id)
        else:
            m += 1

    for F_token in fr:
        fr_id = fr_word_id.get(F_token.lower(), None)
        if fr_id is not None:
            # print("\ncurrent french word: " + F_token.lower())
            # trans is a dictionary
            trans = reverse_table.get(fr_id)
            # sort dictionary by value, reverse order (import operator)
            sorted_trans = sorted(trans.items(), key=operator.itemgetter(1), reverse=True)
            most_lite_tuple = sorted_trans[0]
            # indice, proba
            indice = most_lite_tuple[0]
            top_proba = most_lite_tuple[1]

            # word = list(en_word_id.keys())[list(en_word_id.values()).index(indice)]
            # print("the most literal: ", word, top_proba)
            # above: find the most literal translation of a french word and its top_proba

            group_indice = []
            group_proba = []

            for E_token in eng:
                en_id = en_word_id.get(E_token.lower(), None)
                if en_id is not None:
                    if en_id in trans.keys():
                        en_proba = trans[en_id]
                        if float(en_proba) != 0.0:
                            # print("found translation: " + E_token.lower() + " " + en_proba)
                            group_indice.append(en_id)
                            group_proba.append(en_proba)
                    elif en_id not in trans.keys():
                        # will be counted later
                        # print("\t\t" + E_token + ": probably not a translation")
                        pass
                else:
                    # already count before
                    pass

            # print("Aligned english indice in lexical table: ", group_indice)  # match en word_id
            # print("Aligned english probas in lexical table: ", group_proba)  # their proba

            most_literal += float(top_proba)

            if group_proba:
                max_p = max(group_proba)
                a = group_proba.index(max_p)
                max_word_id = group_indice[a]
                # which english word gets the max proba for current french word
                all_max_en_id.append(max_word_id)

                # comparison of two proba:
                real_trans += float(max_p)
            else:
                k += 1
        else:
            j += 1

    # print("\nmost_literal ", most_literal)
    # print("real_trans ", real_trans)

    delta = most_literal - real_trans
    if most_literal == real_trans == 0.0 :
        print(np.NaN, file=fdelta, end="\t")
        # print("nan")
    else:
        print(delta, file=fdelta, end="\t")
        # print("Delta with most literal: ", delta)

    diff = [x for x in all_en_id if x not in all_max_en_id]
    unaligned_en = len(diff) + m
    unaligned_fr = k + j
    print(unaligned_en/len(eng)*1.0, file=fUnalign, end="\t")
    print(unaligned_fr/len(fr)*1.0, file=fUnalign, end="\t")

    # print("FR words without id (so without most literal translation) ", j)
    # print("FR words without most probable translation ", k)
    # print("EN words without id ", m)
    # print("EN words which are not the most literal translation ", len(diff))
    # print("------\n")

# ------------------------------------------------------------------------------------------------------------

# original segment & lemmatised ones (this method will be executed twice)
# 300 dimension embeddings
def cnet_embedding(eng, fr, eng_content, fr_content, output):
    # consult pickled conceptnet embeddings: numberbatch
    # list of normalized words
    eng_uri = t2u.standardized_uri('en', eng)
    fr_uri = t2u.standardized_uri('fr', fr)

    # ex. eng_uri = /c/en/all_people_in_world

    # if there exists an embedding for the entire segment  (not bad, many for less than 3 words)
    if eng_uri in CNet_enfr_embeddings and fr_uri in CNet_enfr_embeddings:
        en_emb = CNet_enfr_embeddings[eng_uri]
        fr_emb = CNet_enfr_embeddings[fr_uri]
        cosine = 1 - spatial.distance.cosine(en_emb, fr_emb)
        print(cosine, '\t', file=output, end="")
        # print(cosine)
        # print("MWE embedding matched")
    # otherwise, for only content words:
    else:
        en_array = []
        fr_array = []
        for en in eng_content:
            eng_uri = '/c/en/' + en
            if eng_uri in CNet_enfr_embeddings:
                en_emb = CNet_enfr_embeddings[eng_uri]
                en_array.append(en_emb)

        for fr in fr_content:
            fr_uri = '/c/fr/' + fr
            if fr_uri in CNet_enfr_embeddings:
                fr_emb = CNet_enfr_embeddings[fr_uri]
                fr_array.append(fr_emb)

        if en_array and fr_array:     # if they're not empty lists
            en_np_array = np.asarray(en_array, dtype='float32')
            en_mean = np.mean(en_np_array, axis=0, keepdims=True)

            fr_np_array = np.asarray(fr_array, dtype='float32')
            fr_mean = np.mean(fr_np_array, axis=0, keepdims=True)

            cosine = 1 - spatial.distance.cosine(en_mean, fr_mean)
            print(cosine, file=output, end="\t")
            # print(cosine)
        # if no information at all, even for content words list
        else:
            print(np.NaN, file=output, end="\t")
            # print("no info")

#---------------------------------------------------------------------------- feature on constituency parsing

def en_constituent_parsing(enCons, newTabEn):
    i = 0
    hit_line_ids = []
    terminal_labels = []
    cons_en = ""
    type_en = ""
    with open(enCons, "r") as file:
        # after looping "for line in file", the file object is empty
        # so read file into list of lines, we can reuse it later
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            # ex. (IN for)_1
            # had an error: _wordID contains the indice number, but is not the exact one
            if any(re.search(r'_%s\b' % indice, line) for indice in newTabEn):
                # print(line)
                terminal_labels.append(line.split(' ')[0].lstrip('('))
                # note down the lineID
                hit_line_ids.append(i)
            i += 1
        # if there's only one word at one side, take its own terminal node label
        if len(terminal_labels) == 1:
            label = terminal_labels[0]
            label = map_pos_tag(label, 'en')   # map en pos tag to universal ones
            cons_en = label
            type_en = "pos"
        # if there's more than one word at one side, take its non-terminal node label
        else:
            # we take hit_line_ids[0], because words are listed in order in the const file
            first_hit_line = hit_line_ids[0]
            # loop back over the lines before first_hit_line
            # to find the first non terminal node label, before the first word of the segment
            # which is deemed to be the non-terminal label of the whole segment
            for x in range(1, 8):
                string = lines[first_hit_line - x].lstrip(' ').rstrip(' |\n')
                # if see non-terminal node label, stop looping back
                if re.match(r'^[A-Z() ]+$', string):
                    # print(string)
                    # take the last non-terminal label in this line
                    tag = string.split(' ')[-1].lstrip('(')
                    tag = map_constituent_tag(tag, 'en')    # map en constituent tag to a smaller list
                    cons_en = tag
                    type_en = "const"
                    break
        return cons_en, type_en

def fr_constituent_parsing(frCons, newTabFr):
    i = 0
    hit_line_ids = []
    terminal_labels = []
    cons_fr = ""
    type_fr = ""
    with open(frCons, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            if any(re.search(r'_%s\b' % indice, line) for indice in newTabFr):
                # print(line)
                terminal_labels.append(line.split(' ')[0].lstrip('('))
                hit_line_ids.append(i)
            i += 1
        if len(terminal_labels) == 1:
            label = terminal_labels[0]
            label = map_pos_tag(label, 'fr')
            cons_fr = label
            type_fr = "pos"
        else:
            first_hit_line = hit_line_ids[0]
            for x in range(1, 8):
                string = lines[first_hit_line - x].lstrip(' ').rstrip(' |\n')
                # French constituent tag set names contain lowercase, ex Sint
                if re.match(r'^[A-Z() ]+$', string.upper()):
                    tag = string.split(' ')[-1].lstrip('(')
                    tag = map_constituent_tag(tag, 'fr')
                    cons_fr = tag
                    type_fr = "const"
                    break
        return cons_fr, type_fr

#------------------------------------------------------------------------------------------------------- feature on internal dependency parsing

def en_internal_dep(enDep, newTabEn):
    en_rel = []
    rel_inside = False
    with open(enDep, "r") as file:
        for line in file:
            line = line.strip('\n')
            # tackle this kind of problem: nmod:by(clinging-4''', chicanery-14)
            line = re.sub(r"(-\d+)(')+", r'\1', line)
            if any(re.search(r'-%s(,|\))' % indice, line) for indice in newTabEn):
                # print(line)
                m = re.match(r'(.*?)\(.*?-(\d+), .*?-(\d+)\)$', line)
                relation = m.group(1)
                gov = m.group(2)
                dep = m.group(3)
                if int(gov) in newTabEn and int(dep) in newTabEn:  # relation inside the segment
                    en_rel.append(id_dep_rel[map_dep_tag(relation)])   #  normalize tag and turn it to ID
                    rel_inside = True
        if len(newTabEn) == 1:
            en_rel.append(id_dep_rel['lex_norel'])
        else:
            if rel_inside == False:
                en_rel.append(id_dep_rel['seg_norel'])
        dict_en_rel = dict((x, en_rel.count(x)) for x in set(en_rel))  # ex. dic[nmod's ID] = 2 (nb of occurrence)
        # print(dict_en_rel)
        en_rel_vector = [0] * 35  # in total 34 dependency relation tags, see id_dep_rel
        # fill N-hot encoding vector: count presence of each dependency relation tag
        for key in dict_en_rel.keys():
            en_rel_vector[key] = dict_en_rel[key]
        return (en_rel_vector)

def fr_internal_dep(frDep, newTabFr):
    fr_rel = []
    rel_inside = False
    with open(frDep, "r") as file:
        for line in file:
            line = line.strip('\n')
            # tackle this kind of problem: conj:et(limités-3, limités-3''')
            line = re.sub(r"(-\d+)(')+", r'\1', line)
            if any(re.search(r'-%s(,|\))' % indice, line) for indice in newTabFr):
                # print(line)
                m = re.match(r'(.*?)\(.*?-(\d+), .*?-(\d+)\)$', line)
                relation = m.group(1)
                gov = m.group(2)
                dep = m.group(3)
                if int(gov) in newTabFr and int(dep) in newTabFr:  # relation inside the segment
                    # print("rel inside: " + relation)
                    fr_rel.append(id_dep_rel[map_dep_tag(relation)])
                    rel_inside = True
        if len(newTabFr) == 1:
            fr_rel.append(id_dep_rel['lex_norel'])
        else:
            if rel_inside == False:
                fr_rel.append(id_dep_rel['seg_norel'])
        dict_fr_rel = dict((x, fr_rel.count(x)) for x in set(fr_rel))
        # print(dict_fr_rel)
        fr_rel_vector = [0] * 35
        for key in dict_fr_rel.keys():
            fr_rel_vector[key] = dict_fr_rel[key]
        return(fr_rel_vector)

#------------------------------------------------------------------------------------------------------- feature on external dependency parsing

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def en_external_dep(enDep, newTabEn):
    # extract all linked words outside the segment
    en_linked = set()
    # extract the dependency relations with all linked words outside the segment
    # attention: a governor word can have several dependants, so not key:string value, but key:list value
    en_dic_link = {}
    with open(enDep, "r") as file:
        for line in file:
            line = line.strip('\n')
            # remove noise in dep parsing file
            line = re.sub(r"(-\d+)[']+", r'\1', line)
            # indice starts from 1
            if any(re.search(r'-%s(,|\))' % indice, line) for indice in newTabEn):
                # print(line)
                m = re.match(r'(.*?)\(.*?-(\d+), .*?-(\d+)\)$', line)
                # map dep relations to a smaller list
                relation = map_dep_tag(m.group(1))
                gov = m.group(2)
                dep = m.group(3)
                id_gov = int(gov)
                id_dep = int(dep)
                if id_gov not in newTabEn:
                    # linked outside words which are governors
                    en_linked.add(id_gov)
                    if id_gov not in en_dic_link:
                        taglist = []
                        taglist.append(relation)
                        en_dic_link[id_gov] = taglist
                    else:
                        en_dic_link[id_gov].append(relation)
                elif id_dep not in newTabEn:
                    en_linked.add(id_dep)
                    if id_dep not in en_dic_link:
                        taglist = []
                        taglist.append(relation)
                        en_dic_link[id_dep] = taglist
                    else:
                        en_dic_link[id_dep].append(relation)
    return(en_linked, en_dic_link)
    # print("--------------")

def fr_external_dep(frDep, newTabFr):
    fr_linked = set()
    fr_dic_link = {}
    with open(frDep, "r") as file:
        for line in file:
            line = line.strip('\n')
            line = re.sub(r"(-\d+)[']+", r'\1', line)
            if any(re.search(r'-%s(,|\))' % indice, line) for indice in newTabFr):
                # print(line)
                m = re.match(r'(.*?)\(.*?-(\d+), .*?-(\d+)\)$', line)
                relation = map_dep_tag(m.group(1))
                gov = m.group(2)
                dep = m.group(3)
                id_gov = int(gov)
                id_dep = int(dep)
                if id_gov not in newTabFr:
                    fr_linked.add(id_gov)
                    if id_gov not in fr_dic_link:
                        taglist = []
                        taglist.append(relation)
                        fr_dic_link[id_gov] = taglist
                    else:
                        fr_dic_link[id_gov].append(relation)
                elif id_dep not in newTabFr:
                    fr_linked.add(id_dep)
                    if id_dep not in fr_dic_link:
                        taglist = []
                        taglist.append(relation)
                        fr_dic_link[id_dep] = taglist
                    else:
                        fr_dic_link[id_dep].append(relation)
    return(fr_linked, fr_dic_link)
    # print("--------------")

def compare_external_dep(en_linked, fr_linked, en_dic_link, fr_dic_link, aligns, fileID):
    # alignment file: indice starts from 0. fileID starts from 1
    # fileID-1 corresponds to which line in alignment file
    alignment = aligns[fileID - 1]
    # the first element is the sentenceID
    align_points = alignment.split(' ')[1:]
    # print(align_points)

    # turn from dep parsing to alignment, -1
    en_linked = [int(x) - 1 for x in en_linked]
    fr_linked = [int(x) - 1 for x in fr_linked]

    en_link_aligned = []
    fr_link_aligned = []
    for item in align_points:
        a = item.split(':')[0]
        b = item.split(':')[1]
        if a != '' and b != '':  # avoid unaligned cases
            # trans_process = item.split(':')[2]
            lista = a.split(',')
            listb = b.split(',')
            lista = [int(x) for x in lista]
            listb = [int(x) for x in listb]
            # find aligned pairs among the linked outside words
            if any(x in lista for x in en_linked) and any(y in listb for y in fr_linked):
                # print("alignments containing linked outside words: ", item)
                en_link_aligned.extend(intersection(lista, en_linked))  # extend: add all elements of a list
                fr_link_aligned.extend(intersection(listb, fr_linked))

    # indice starts from 1
    # en_dic_link[a linked outside word indice]=[list of dep tags associated]
    # print(en_dic_link)
    # print(fr_dic_link)

    # +1 turn from alignment to dependency
    en_link_aligned = [x + 1 for x in en_link_aligned]
    fr_link_aligned = [x + 1 for x in fr_link_aligned]

    # print(en_link_aligned)
    # print(fr_link_aligned)

    final_en_rel = []
    for x in en_link_aligned:
        dep_tag = en_dic_link[x]
        final_en_rel.extend(dep_tag)
    # print(final_en_rel)  # the list (not set) of relations tags, to be compared
    # turn dep tag to id representation
    final_en_rel = [id_dep_rel[x] for x in final_en_rel]

    en_count_rel = dict((x, final_en_rel.count(x)) for x in set(final_en_rel))
    en_rel_vector = [0] * 33  # in total 33 dependency relation tags, without 'lex|seg_no_rel'
    for key in en_count_rel.keys():
        en_rel_vector[key] = en_count_rel[key]
    # print(en_rel_vector)
    #---------------------------------

    final_fr_rel = []
    for x in fr_link_aligned:
        dep_tag = fr_dic_link[x]
        final_fr_rel.extend(dep_tag)
    # print(final_fr_rel)
    final_fr_rel = [id_dep_rel[x] for x in final_fr_rel]

    fr_count_rel = dict((x, final_fr_rel.count(x)) for x in set(final_fr_rel))
    fr_rel_vector = [0] * 33
    for key in fr_count_rel.keys():
        fr_rel_vector[key] = fr_count_rel[key]
    # print(fr_rel_vector)

    return(en_rel_vector, fr_rel_vector)
#-------------------------------------------------------------------------------------------------------


def check_pos_pattern(eng_pos, fr_pos):
    # * coal -> charbonneuse
    if len(eng_pos.split(' ')) == len(fr_pos.split(' ')) == 1:
        if eng_pos != fr_pos:
            # return ("1")    avoid cumulating errors from pos tagging
            return ("0")
    else:
        ### strong rule:
        # either or -> ou
        if 'ADV CCONJ' == eng_pos and 'CCONJ' == fr_pos:
            return ("1")
        # which is -> soit
        elif 'DET VERB' == eng_pos and 'CCONJ' == fr_pos:
            return ("1")
        # or so -> près d'
        elif 'CCONJ ADV' == eng_pos and 'ADV ADP' == fr_pos:
            return ("1")

        # as is # comme
        elif 'ADP VERB' == eng_pos and 'SCONJ' == fr_pos:
            return ("1")
        # * after -> au bout de; into the -> au sein du
        elif re.match(r'^ADP( DET)?$', eng_pos) and 'ADP NOUN ADP' == fr_pos:
            return ("1")
        # so -> pour que
        elif 'ADP' == eng_pos and 'ADP CCONJ' == fr_pos:
            return ("1")
        # from -> produit par
        elif 'ADP' == eng_pos and 'VERB ADP' == fr_pos:
            return ("1")
        # around -> situé autour de
        elif 'ADP' == eng_pos and 'ADJ ADV ADP' in fr_pos:
            return ("1")
        # around -> qui entoure; of -> ce qui parle de
        elif 'ADP' == eng_pos and 'PRON VERB' in fr_pos:
            return ("1")
        # as a result -> il en résulte qu'
        elif 'ADP DET NOUN' in eng_pos and 'PRON VERB SCONJ' in fr_pos:
            return ("1")
        # in a nutshell -> pour résumer
        elif 'ADP DET NOUN' == eng_pos and 'ADP VERB' == fr_pos:
            return ("1")
        # * in winter -> hivernales
        elif 'ADP NOUN' in eng_pos and 'ADJ' == fr_pos:
            return ("1")
        # in this simple way -> simplement de cette manière
        # in the exact -> exactement de la
        elif 'ADP DET ADJ' in eng_pos and 'ADV ADP DET' in fr_pos:
            return ("1")
        # while it plays -> pendant la lecture
        elif 'ADP PRON VERB' in eng_pos and 'ADP DET NOUN' in fr_pos:
            return ("1")
        # so they can -> de manière à pouvoir
        elif 'ADP PRON AUX' == eng_pos and 'ADP NOUN' in fr_pos:
            return ("1")

        # * all of # tous/toutes
        elif 'DET ADP' == eng_pos and 'ADJ' == fr_pos:
            return ("1")
        # spend a large sum of money -> dépenser massivement
        elif re.search(r'DET ADJ NOUN', eng_pos) and 'VERB ADV' == fr_pos:
            return ("1")
        # in this simple way -> simplement de cette manière
        elif 'DET ADJ NOUN' in eng_pos and 'ADV ADP DET NOUN' == fr_pos:
            return ("1")
        # make a little pierce -> percer un peu
        # are a better match to -> correspondent mieux à
        elif re.search('DET ADJ NOUN', eng_pos) and re.search(r'VERB (DET )?ADV', fr_pos):
            return ("1")

        # treacherous -> la traîtrise des; spatial -> dans l' espace
        # open-ended -> avec une fin ouverte; deceptive -> une illusion
        elif 'ADJ' == eng_pos and re.search(r'(ADP )?DET NOUN( ADP|ADJ)?', fr_pos):
            return ("1")
        # * everyday -> du quotidien; rugged -> en haillons; professional -> de professionnels
        elif 'ADJ' == eng_pos and 'ADP NOUN' == fr_pos:
            return ("1")
        # next -> à venir
        elif 'ADJ' == eng_pos and 'ADP VERB' == fr_pos:
            return ("1")
        # western china -> l' ouest de la chine
        elif 'ADJ NOUN' == eng_pos and 'NOUN ADP DET NOUN' in fr_pos:
            return ("1")
        # foreign -> que nous importons
        elif re.search(r'ADJ( ADP)?', eng_pos) and 'PRON VERB' in fr_pos:
            return ("1")
        # human-computer -> entre humains et ordinateurs
        elif 'ADJ' == eng_pos and re.search(r'ADP NOUN CCONJ NOUN', fr_pos):
            return ("1")
        # loud -> à haute voix, healthy -> en bonne santé
        elif 'ADJ' == eng_pos and re.search(r'^ADP ADJ NOUN$', fr_pos):
            return  ("1")
        # deep ocean -> fond des océans
        elif 'ADJ' in eng_pos and re.search(r'^NOUN ADP NOUN$', fr_pos):
            return ("1")

        # where -> l' endroit où, how -> la façon dont
        elif 'ADV' == eng_pos and re.search(r'DET NOUN PRON', fr_pos):
            return ("1")
        # just -> n' rien qu'
        elif 'ADV' == eng_pos and re.search(r'^PART ADV', fr_pos):
            return ("1")
        # how -> de quelle façon
        elif 'ADV' == eng_pos and 'ADP DET NOUN' == fr_pos:
            return ("1")
        # only -> ne que
        elif 'ADV' == eng_pos and 'PART SCONJ' == fr_pos:
            return ("1")
        # * methodologically -> de façon méthodologique
        elif 'ADV' == eng_pos and re.search(r'^ADP NOUN ADJ$', fr_pos):
            return ("1")
        # hopefully -> en espérant
        elif 'ADV' == eng_pos and re.search(r'ADP VERB', fr_pos):
            return ("1")
        # how we understand -> notre façon de comprendre
        # how we -> notre façon de
        elif re.search(r'ADV PRON( VERB)?', eng_pos) and re.search(r'DET NOUN( ADP)?( VERB)?', fr_pos):
            return ("1")
        # how to -> la manière de
        elif re.search('ADV ADP', eng_pos) and re.search(r'DET NOUN ADP', fr_pos):
            return ("1")
        # how -> des raisons pour lesquelles; why -> pour laquelle
        elif 'ADV' == eng_pos and re.search(r'(NOUN )?ADP PRON', fr_pos):
            return ("1")
        # how it unfolds -> le dénouement
        elif 'ADV PRON VERB' == eng_pos and re.match(r'^(ADP|DET) NOUN$', fr_pos):
            return ("1")
        # how big are -> la taille de
        elif 'ADV ADJ VERB' == eng_pos and 'DET NOUN ADP' in fr_pos:
            return ("1")

        # treatment -> se soigner; the source was -> elle venait d'
        elif re.search(r'NOUN( VERB)?', eng_pos) and 'PRON VERB' in fr_pos:
            return ("1")
        # sketch -> font des croquis
        elif 'NOUN' == eng_pos and 'VERB DET NOUN' in fr_pos:
            return ("1")
        # * children 's -> pour enfants
        elif 'NOUN PART' == eng_pos and 'ADP NOUN' == fr_pos:
            return ("1")
        # storytelling institution -> institution de contes
        elif 'NOUN NOUN' == eng_pos and re.search('NOUN ADP( DET)? NOUN', fr_pos):
            return ("1")
        # vision impairment -> l' altération visuelle
        elif 'NOUN NOUN' == eng_pos and 'DET NOUN ADJ' == fr_pos:
            return ("1")

        # reading -> la lecture de
        elif 'VERB' == eng_pos and re.match(r'^(ADP|DET) NOUN', fr_pos):
            return ("1")
        # * 'd be able to -> pourriez; are interested in -> s' intéressent à
        # living homeless -> sans-abri
        elif re.search(r'VERB ADJ( ADP)?', eng_pos) and re.search(r'(PRON )?VERB( ADP)?', fr_pos):
            return ("1")
        # is meaningful -> a un sens
        elif re.search(r'VERB ADJ', eng_pos) and re.search(r'VERB DET NOUN', fr_pos):
            return ("1")
        # linked together -> liées entre elles; grasping -> prenant en main
        # are increasing rapidly -> est en augmentation rapide
        elif re.search(r'VERB( ADV)?', eng_pos) and re.search(r'VERB ADP (PRON|NOUN)( ADJ)?', fr_pos):
            return ("1")
        # extending westward -> vers l' ouest
        elif 'VERB ADV' == eng_pos and 'ADP DET NOUN' == fr_pos:
            return ("1")
        # is a problem -> est problématique
        elif 'VERB DET NOUN' in eng_pos and 'VERB ADJ' in fr_pos:
            return ("1")
        # became the basis for -> ont inspiré
        elif 'VERB DET NOUN ADP' in eng_pos and 'AUX VERB' in fr_pos:
            return ("1")
        # related to -> avait un lien avec; acting as -> dans le rôle de
        # looking for -> faisant des recherches pour
        elif 'VERB ADP' == eng_pos and re.search(r'(VERB|ADP) DET NOUN ADP', fr_pos):
            return ("1")
        # colliding with -> est entré en collision avec
        elif 'VERB ADP' == eng_pos and re.search(r'VERB ADP NOUN ADP', fr_pos):
            return ("1")
        else:
            return ("0")

#----------------------------------------------------------------------------------------

# ewords, fwords are tables of filtered lemmatised words
def link_percent(ewords, fwords):
    fr_uniq = set()
    en_uniq = set()

    for eword in ewords:
        for fword in fwords:
            # print("--- current pair: ", eword, ' # ', fword, '\n')
            bridges = set()

            output = link(eword, fword, en_fr_assert_direct, en_fr_assert_reverse)
            if output and output != {1} and output != {2} and output != {3}:
                bridges = output

            i = 0
            if bridges:
                for bridge in bridges:
                    hit = link(bridge, fword, fr_fr_assert_direct, fr_fr_assert_reverse)
                    if hit == {1}:
                        # print('The French bridge word \'' + bridge + '\' is linked to target ' + fword)
                        i += 1
                    else:
                        pass
                if i != 0:
                    # print("This pair is linked via " + str(i) + " French bridge word(s).")
                    fr_uniq.add(fword)
                    en_uniq.add(eword)
                else:
                    tmp = Levenshtein.distance(eword, fword)
                    if tmp == 0:
                        fr_uniq.add(fword)
                        en_uniq.add(eword)
                    else:
                        pass

    # print("unique indirectly linked words, or with Levenshtein distance 0: ", fr_uniq, en_uniq)
    percent = (len(fr_uniq) + len(en_uniq)) / (len(set(ewords)) + len(set(fwords)))
    return percent


# eseg, fseg are strings!
# if there are several words, they are connected with "_"
# strings are tested under three forms: original, lemmatized, lemmatized + filtered
def entire_seg_link(eseg, fseg):
    # this method will return either 1 or 0
    # print(eseg, fseg)
    output = link(eseg, fseg, en_fr_assert_direct, en_fr_assert_reverse)
    # print(output)

    bridges = set()

    if output == {1}:
        # print("This pair is directly linked in CNet")
        return 0   # means directly linked
    elif output == {2}:
        # print("The start word doesn't exist in CNet resource")
        pass
    elif output == {3}:
        # print("This pair is not directly linked in CNet, and no bridge words found")
        pass
    else:
        # print("Not directly linked, but we find these potential bridge words: ", output)
        # first check en-fr pair in en-fr assertions
        # 1) directly linked 2) en segment not present 3) not directly linked, found other bridge words
        bridges = output

    # exploit bridge words
    i = 0
    if bridges:
        for bridge in bridges:
            # see if fr_bridge and fr_target are linked
            a = link(bridge, fseg, fr_fr_assert_direct, fr_fr_assert_reverse)
            if a == {1}:      # target word is directly linked with bridge word
                # print('\tThe French bridge word \'' + bridge + '\' is directly linked to ' + fseg)
                i += 1
            else:
                pass
        if i != 0:
            # print("This pair is linked via " + str(i) + " French bridge word(s).")
            return 1   # means indirectly linked via French bridge words
        elif i == 0:
            # print("This pair is not linked, even via French bridge words")
            pass

# in ConceptNet assertions file, check if two words/segments are:
# directly linked in assertions file
# or not present in resource
# or not directly linked, but we (can/can't) find possible bridge words
def link(start, end, direct_dic, reverse_dic):
    # print("search link for ", start, " : ", end)
    results = set()   # this set will receive the possible target that we want + the other 'bridge' words

    # if direct_dic == en_fr_assert_direct :
    #    a = "bi"
    # else:
    #    a = "mono"
    # print('\nsearch {} in '.format(start) + a + 'lingual-assertions\n')

    total_flag = False
    dir_exis_flag = True
    rev_exis_flag = True

    dir_flag = False
    if direct_dic.get(start) != None:
        value = direct_dic[start]
        ## 'value' is a list of 'triple' strings for each start query: end + "#" + relation + "#" + weight
        for triple in value:
            results.add(triple)
            if end == triple.split('#')[0]:
                dir_flag = True
                total_flag = True
                # print(start, triple)
        # if dir_flag == False:
        #     print(end + " is not linked with " + start + " in direct dictionary")
        # else:
        #     print(end + " is linked with " + start + " in direct dictionary")
    else:
        # print(start + " isn't in direct dictionary")
        dir_exis_flag = False

    rev_flag = False
    if reverse_dic.get(start) != None:
        value = reverse_dic[start]
        for triple in value:
            results.add(triple)
            if end == triple.split('#')[0]:
                rev_flag = True
                total_flag = True
                # print(start, triple)
        # if rev_flag == False:
        #     print(end + " is not linked with " + start + " in reverse dictionary")
        # else:
            # print(end + " is linked with " + start + " in reverse dictionary")
    else:
        # print(start + " isn't in reverse dictionary")
        rev_exis_flag = False

    if total_flag == True:
        # print("==>" + start + " is directly linked with " + end)
        return set([1])
    else:
        if dir_exis_flag == rev_exis_flag == False:
            # print("==>" + start + " isn't in direct/reverse dictionary")
            return set([2])
        else:
            # print("==>" + start + " isn't directly linked with " + end)
            bridge_word_set = set()
            for triple in results:
                bridge_word_set.add(triple.split('#')[0])
            if bridge_word_set:
                # print("potential bridge words: ", bridge_word_set)
                return bridge_word_set
            else:
                # print("no bridge word found. but actually:
                # 1) en segment exists, linked with our target + possible bridges
                # 2) en segment doesn't exist at all")
                return set([3])


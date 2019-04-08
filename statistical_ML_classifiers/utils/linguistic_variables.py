"""
author: Yuming ZHAI
"""

en_univ_posTag = {
',':'PUNCT',
'.':'PUNCT',
':':'PUNCT',
'``':'PUNCT',
'(':'PUNCT',
')':'PUNCT',
'-LRB-':'PUNCT',
'-RRB-':'PUNCT',
'HYPH':'PUNCT',
'"':'PUNCT',
'CC':'CCONJ',
'CD':'NUM',
'DT':'DET',
'EX':'PRON',
'FW':'X',
'IN':'ADP',
'JJ':'ADJ',
'JJR':'ADJ',
'JJS':'ADJ',
'LS':'X',
'MD':'AUX',
'NN':'NOUN',
'NNP':'PROPN',
'NNPS':'PROPN',
'NNS':'NOUN',
'PDT':'DET',
'POS':'PART',
'PRP$':'PRON',
'PRP':'PRON',
'RB':'ADV',
'RBR':'ADV',
'RBS':'ADV',
'RP':'ADP',
'SYM':'SYM',
'TO':'ADP',
'UH':'INTJ',
'VB':'VERB',
'VBD':'VERB',
'VBG':'VERB',
'VBN':'VERB',
'VBP':'VERB',
'VBZ':'VERB',
'WDT':'DET',
'WP$':'PRON',
'WP':'PRON',
'WRB':'ADV'
}

fr_univ_posTag = {
'ADJ':'ADJ',
'ADJWH':'ADJ',
'ADV':'ADV',
'ADVWH':'ADV',
'C':'CCONJ',
'CC':'CCONJ',
'CL':'PRON',
'CLO':'PRON',
'CLR':'PRON',
'CLS':'PRON',
'CS':'SCONJ',
'DET':'DET',
'DETWH':'DET',
'ET':'X',      # foreign words
'I':'INTJ',
'N':'NOUN',
'NC':'NOUN',
'NPP':'PROPN',
'P':'ADP',
'P+D':'ADP',    # preposition + determinant
'P+PRO':'ADP',
'PREF':'X',
'PRO':'PRON',
'PROREL':'PRON',
'PROWH':'PRON',
'PUNC':'PUNCT',
'PONCT':'PUNCT',    # bonsai constituent parsing uses this tag
'V':'VERB',
'VIMP':'VERB',
'VINF':'VERB',
'VPP':'VERB',
'VPR':'VERB',
'VS':'VERB',
# here complete universal to universal mapping, because Bonsai couldn't parse a too long sentence
# so I replace it by corenlp's parsing output
'ADP':'ADP',
'AUX':'AUX',
'CCONJ':'CCONJ',
'INTJ':'INTJ',
'NOUN':'NOUN',
'NUM':'NUM',
'PART':'PART',
'PRON':'PRON',
'PROPN':'PROPN',
'PUNCT':'PUNCT',
'SCONJ':'SCONJ',   # subordinating conjunction
'SYM':'SYM',
'VERB':'VERB',
'X':'X'
}

# some decisions are taken according to (Leung et al.,2016)
zh_univ_posTag = {
'AD':'ADV',
'AS':'PART',
'BA':'ADP',
'CC':'CCONJ',
'CD':'NUM',
'CS':'SCONJ',
'DEC':'PART',
'DEG':'PART',
'DER':'PART',
'DEV':'PART',
'DT':'DET',
'ETC':'CCONJ',
'IJ':'INTJ',
'JJ':'ADJ',
'LC':'ADP',
'M':'NOUN',
'MSP':'CCONJ',
'NN':'NOUN',
'NR':'PROPN',
'NT':'NOUN',
'OD':'ADJ',
'P':'ADP',
'PN':'PRON',
'PU':'PUNCT',
'SB':'AUX',
'SP':'PART',
'VA':'ADJ',
'VC':'AUX',
'VE':'VERB',
'VV':'VERB'
}

id_univ_postag = {
'ADJ':0,
'ADP':1,
'ADV':2,
'AUX':3,
'CCONJ':4,
'DET':5,
'INTJ':6,
'NOUN':7,
'NUM':8,
'PART':9,
'PRON':10,
'PROPN':11,
'PUNCT':12,
'SCONJ':13,   # subordinating conjunction
'SYM':14,
'VERB':15,
'X':16    # other
}

en_const_Tag = {
'ROOT':'ROOT',
'ADJP':'AP',
'ADVP':'ADVP',
'CONJP':'CONJP',
'PP':'PP',
'NP':'NP',
'VP':'VP',
'S':'CLAUSE',
'SBAR':'CLAUSE',
'SBARQ':'CLAUSE',
'SINV':'CLAUSE',
'SQ':'CLAUSE',
'FRAG':'x',  # use 'x' to ignore these tags
'INTJ':'x',
'LST':'x',   # List marker. Includes surrounding punctuation.
'NAC':'x',   # Not a Constituent; used to show the scope of certain prenominal modifiers within an NP.
'NX':'x',    # Used within certain complex NPs to mark the head of the NP. Corresponds very roughly to N-bar level but used quite differently.
'PRN':'x',   # Parenthetical.
'PRT':'x',   # Particle. Category for words that should be tagged RP.
'QP':'x',    # Quantifier Phrase (i.e. complex measure/amount phrase); used within NP.
'RRC':'x',   # Reduced Relative Clause.
'UCP':'x',   # Unlike Coordinated Phrase.
'X':'x',     # Unknown, uncertain, or unbracketable. X is often used for bracketing typos and in bracketing the...the-constructions.
'WHADJP':'AP',   # Wh-adjective Phrase. Adjectival phrase containing a wh-adverb, as in how hot.
'WHADVP':'ADVP',  # Wh-adverb Phrase. Introduces a clause with an NP gap. May be null (containing the 0 complementizer) or lexical, containing a wh-adverb such as how or why.
'WHNP':'NP',     # Wh-noun Phrase. Introduces a clause with an NP gap. May be null (containing the 0 complementizer) or lexical, containing some wh-word, e.g. who, which book, whose daughter, none of which, or how many leopards.
'WHPP':'PP'      # Wh-prepositional Phrase. Prepositional phrase containing a wh-noun phrase (such as of which or by whose authority) that either introduces a PP gap or is contained by a WHNP.
}

fr_const_Tag = {
'SENT':'ROOT',
'COORD':'CONJP',
'AP':'AP',
'NP':'NP',
'AdP':'ADVP',
'VN':'VP',
'VPinf':'VP',
'VPpart':'VP',
'PP':'PP',
'Sint':'CLAUSE',   # finite clause
'Srel':'CLAUSE',
'Ssub':'CLAUSE'
}

id_const_Tag = {
'ROOT':0,
'PP':1,
'AP':2,
'CONJP':3,
'NP':4,
'ADVP':5,
'VP':6,
'CLAUSE':7,
'x':8
}

# map pos to a const, to see whether they belongs to the same category
map_pos_const = {
'ADJ':'AP',
'ADP':'PP',
'ADV':'ADVP',
'NOUN':'NP',
'NUM':'NP',
'DET':'NP',
'PRON':'NP',
'PROPN':'NP',
'CCONJ':'CONJP',
'VERB':'VP',
'SCONJ':'CLAUSE',
'X':'x',
'AUX':'x',
'PART':'x',
'SYM':'x',
'INTJ':'x'
}

# enchanced plus plus dependencies
en_fr_dep_rel = {
'acl':'acl',
'advcl':'advcl',
'advmod':'advmod',
'amod':'amod',
'appos':'appos',
'aux':'aux',
'auxpass':'auxpass',
'case':'case',
'cc':'cc',
'ccomp':'ccomp',
'compound':'compound',
'conj':'conj',
'cop':'cop',
'csubj':'csubj',
'dep':'dep',
'det':'det',
'discourse':'discourse',
'dobj':'dobj',
'expl':'expl',
'iobj':'iobj',
'mark':'mark',
'mwe':'mwe',
'name':'name',
'neg':'neg',
'nmod':'nmod',
'nsubj':'nsubj',
'nsubjpass':'nsubjpass',
'nummod':'nummod',
'punct':'punct',
'parataxis':'parataxis',
'ref':'ref',
'root':'root',
'xcomp':'xcomp'
}

id_dep_rel = {
'acl':0,
'advcl':1,
'advmod':2,
'amod':3,
'appos':4,
'aux':5,
'auxpass':6,
'case':7,
'cc':8,
'ccomp':9,
'compound':10,
'conj':11,
'cop':12,
'csubj':13,
'dep':14,
'det':15,
'discourse':16,
'dobj':17,
'expl':18,
'iobj':19,
'mark':20,
'mwe':21,
'name':22,
'neg':23,
'nmod':24,
'nsubj':25,
'nsubjpass':26,
'nummod':27,
'punct':28,
'parataxis':29,
'ref':30,
'root':31,
'xcomp':32,
'lex_norel':33,    # un seul mot, pas de dépendance interne
'seg_norel':34     # un segment, mais pas de dépendance interne
}
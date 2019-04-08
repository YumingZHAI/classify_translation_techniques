"""
author: Yuming ZHAI

This Python module provides just the code from the 'conceptnet5' module that
you need to represent terms, possibly with multiple words, as ConceptNet URIs.

It depends on 'wordfreq', a Python 3 library, so it can tokenize multilingual
text consistently: https://pypi.org/project/wordfreq/

Example:

>>> standardized_uri('es', 'ayudar')
'/c/es/ayudar'
>>> standardized_uri('en', 'a test phrase')
'/c/en/test_phrase'
>>> standardized_uri('en', '24 hours')
'/c/en/##_hours'

"""
import wordfreq
import re

EN_STOPWORDS = ['the', 'a', 'an']   # so not totally the same thing as my "en_content"
FR_STOPWORDS = []

EN_DROP_FIRST = ['to']
# l' : needed in e.g. last_year  #  l_année_dernière
FR_DROP_FIRST = ['le', 'la', 'les', 'des', 'du', 'une', 'un', 'ces']
FR_DROP_FIRST_TWO = ['de la', 'de l']
# they are not at the beginning
EN_sub_dic = {'s': 'is'}  # {'re':'are', 'm':'am', 'd':'would'}
FR_sub_dic = {'qu': 'que', 'd': 'de', 'des': 'de', 'du': 'de'}

DOUBLE_DIGIT_RE = re.compile(r'[0-9][0-9]')
DIGIT_RE = re.compile(r'[0-9]')


# 1
def standardized_uri(language, term):
    """
    Get a URI that is suitable to label a row of a vector space, by making sure
    that both ConceptNet's and word2vec's normalizations are applied to it.

    'language' should be a BCP 47 language code, such as 'en' for English.

    If the term already looks like a ConceptNet URI, it will only have its
    sequences of digits replaced by #. Otherwise, it will be turned into a
    ConceptNet URI in the given language, and then have its sequences of digits
    replaced.
    """
    if not (term.startswith('/') and term.count('/') >= 2):
        term = _standardized_concept_uri(language, term)
    return replace_numbers(term)


# 2
def _standardized_concept_uri(language, term):
    # english_filter and french_filter: has many my own coding
    if language == 'en':
        token_filter = english_filter
    elif language == 'fr':
        token_filter = french_filter
    else:
        token_filter = None
    language = language.lower()
    norm_text = _standardized_text(term, token_filter)
    return '/c/{}/{}'.format(language, norm_text)


# 3 filter on tokenized tokens
def french_filter(tokens):
    non_stopwords = [token for token in tokens if token not in FR_STOPWORDS]
    if non_stopwords and non_stopwords[0] in FR_DROP_FIRST:
        non_stopwords = non_stopwords[1:]
    elif len(non_stopwords) > 2:
        FIRST_TWO = non_stopwords[0] + " " + non_stopwords[1]
        if FIRST_TWO in FR_DROP_FIRST_TWO:
            non_stopwords = non_stopwords[2:]

    # attention: test whether non_stopwords is empty
    if non_stopwords and non_stopwords[-1] in FR_sub_dic.keys():
        non_stopwords[-1] = FR_sub_dic[non_stopwords[-1]]

    if non_stopwords:
        return non_stopwords
    else:
        return tokens


# 3
def english_filter(tokens):
    """
    Given a list of tokens, remove a small list of English stopwords. This
    helps to work with previous versions of ConceptNet, which often provided
    phrases such as 'an apple' and assumed they would be standardized to
	'apple'.
    """
    # filter stop words
    non_stopwords = [token for token in tokens if token not in EN_STOPWORDS]
    # don't keep the first word in list: [to,...]
    if non_stopwords and non_stopwords[0] in EN_DROP_FIRST:
        non_stopwords = non_stopwords[1:]

    # check if two lists have common items: fast
    # sub = bool(set(EN_sub_dic.keys()) & set(non_stopwords))
    if non_stopwords and non_stopwords[-1] in EN_sub_dic.keys():
        non_stopwords[-1] = EN_sub_dic[non_stopwords[-1]]

    if non_stopwords:
        return non_stopwords
    else:
        # if the segment is empty after filtering, just return the given segment
        return tokens


# 4
def _standardized_text(text, token_filter):
    tokens = simple_tokenize(text.replace('_', ' '))
    # first tokenize, then filter
    if token_filter is not None:
        tokens = token_filter(tokens)
    return '_'.join(tokens)


# 5
def simple_tokenize(text):
    """
    Tokenize text using the default wordfreq rules.
    It depends on 'wordfreq', a Python 3 library, so it can tokenize multilingual
    text consistently: https://pypi.org/project/wordfreq/
    """
    return wordfreq.tokenize(text, 'xx')


# 6
# DOUBLE_DIGIT_RE = re.compile(r'[0-9][0-9]')
# DIGIT_RE = re.compile(r'[0-9]')
def replace_numbers(s):
    """
    Replace digits with # in any term where a sequence of two digits appears.
    [This operation is applied to text that passes through word2vec, so we
        should match it.]
    """
    if DOUBLE_DIGIT_RE.search(s):
        return DIGIT_RE.sub('#', s)
    else:
        return s

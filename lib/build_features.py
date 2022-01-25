"""
Filename: build_features.py
Author: Yiming Xia
"""

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from collections import defaultdict

def remove_stopwords(word_tokenized):
    '''Remove stop words.'''
    return [word for word in word_tokenized if not word.lower() in stopwords.words('english')]

def lemmatize_sentence(word_tokenized):
    '''Lemmatize tokenized sentence.'''
    '''Create a '''
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lmtzr = WordNetLemmatizer()
    return [lmtzr.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(word_tokenized)]

def clean_sentence(word_tokenized):
    """Wrapper that removes stop words and lemmatizes tokenized string."""
    tokens_no_stop_words = remove_stopwords(word_tokenized)
    tokens_lemmatized = lemmatize_sentence(tokens_no_stop_words)
    return " ".join(tokens_lemmatized)


def get_vectorized_sentences(lst_sentences):
    """ Maps list of strings to a vector space using a count vectorizer.
    :param (list) lst_sentences: Each element is either a string word/sentence.
    :return (np.array):
        Sparse matrix with number of columns equivalent to number of unique
        words across lst_sentences. For example, A_{ij} represents the number
        of occurrences of word j in the ith element of lst_sentences.
    """
    vectorizer = TfidfVectorizer().fit(lst_sentences)
    return vectorizer.transform(lst_sentences)

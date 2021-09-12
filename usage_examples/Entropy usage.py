#-*- coding: utf-8 -*-

from Entropy import *
from Corpus_preprocessing import general_dictionary

# User should use:
# general_dictionary(corpus)
# # Input: corpus --> list of space separated documents of ALL categories
# # corpus --> list of strings ['sentence is sentence', 'sentence 2']

# entropy_list(general_dict, documents)
# max_entropy(corpus)
# make_pos_neg_keywords_alpha(alpha, pos_entropies, neg_entropies, general_dict)

corpus = [
    'this is a brown apple and smell bad',
    'i love this tasty red apple',
    'this apple is small and has no taste',
    'i love when the apple is juicy',
    'dry apple has no value',
    'sweet apple is best apple',
    'sweet sweet sweet apple apple apple yes yes yes',
    'i like sweet apple with red and green'
]

general_dict = general_dictionary(corpus)

sentences = [
    ['this is a brown apple and smell bad', 0],
    ['i love this tasty red apple', 1],
    ['this apple is small and has no taste', 0],
    ['i love when the apple is juicy', 1],
    ['dry apple has no value', 0],
    ['sweet apple is best apple', 1],
    ['sweet sweet sweet apple apple apple yes yes yes', 1],
    ['i like sweet apple with red and green', 1]
]

positive_documents = [doc[0] for doc in sentences if doc[1]==1]
# ['i love this tasty red apple', 
# 'i love when the apple is juicy', 
# 'sweet apple is best apple', 
# 'sweet sweet sweet apple apple apple yes yes yes', 
# 'i like sweet apple with red and green']

negative_documents = [doc[0] for doc in sentences if doc[1]==0]
# ['this is a brown apple and smell bad', 
# 'this apple is small and has no taste', 
# 'dry apple has no value']

pos_raw_entropies = entropy_list(general_dict, positive_documents)
pos_entropies = [pe/max_entropy(positive_documents) for pe in pos_raw_entropies]
neg_raw_entropies = entropy_list(general_dict, negative_documents)
neg_entropies = [ne/max_entropy(negative_documents) for ne in neg_raw_entropies]

alpha = 2

pos_keys, neg_keys = make_pos_neg_keywords_alpha(alpha, pos_entropies, neg_entropies, general_dict)

# >>> pos_keys
# ['love', 'red', 'sweet']
# >>> neg_keys
# ['and', 'has', 'no', 'this']



#-*- coding: utf-8 -*-

from libraries.SVM_Methods import *
from libraries.BagOfWords import *

k = 4
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
keyword_list = ['brown','red','apple','smell','green','juicy', 'tasty','dry','sweet','yes','taste','love']

def SVM_KFolds_sentences(sentences, k, keyword_list, kernel='linear', C=1.0, gamma=0.001):
    x, y = Vectorize_Bag_of_Words(sentences, keyword_list)
    clf, results = SVM_Kfolds(x, y, k, kernel=kernel, C=C, gamma=gamma)
    return results


SVM_KFolds_sentences(sentences, k, keyword_list, kernel = 'linear', C = 1.0, gamma = 0.001, times = 1)

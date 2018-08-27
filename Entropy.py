#-*- coding: utf-8 -*-

import numpy
import scipy.stats
import scipy.special
import gensim

def Tokenize(corpus):
    tokenized = [i.split() for i in corpus]
    counts = dict()
    for sentence in tokenized:
        for word in sentence:
            if counts.get(word):
                counts[word] += 1
            else:
                counts[word] = 1
    tokenized = [[word for word in sentence if counts.get(word)>1] for sentence in tokenized]
    return tokenized

def Dictionary(tokenized):
    dictionary = gensim.corpora.Dictionary(tokenized)
    # dictionary.save(MakeLogFile("gensimdictionary.dict"))
    return dictionary

# Input: corpus --> list of space separated documents of ALL categories
# corpus --> list of strings ['sentence is sentence', 'sentence 2']
def general_dictionary(corpus):
    tokenized = Tokenize(corpus)
    dictionary = Dictionary(tokenized)
    general_dict = sorted([word for word in dictionary.values()])
    return general_dict

# For a particular word in either category (positive or negative, etc.)
# The following methods should be used in a list of documents of ONLY positive or ONLY negative documents

# Format is list of documents
def getNs(word, documents):
    Ns = []
    for sentence in documents:
        N = sentence.count(word)
        Ns.append(N)
    return Ns

def get_SumN(word, documents):
    SumN = sum(getNs(word,documents))
    return SumN

def probability(N, SumN):
	P = 0
	if SumN>0:
		P = N/SumN
	return P

def P_list(Ns, SumN):
    Ps = []
    for i in Ns:
        P = probability(i, SumN)
        if P>0:
            Ps.append(P)
    return Ps

def entropy(Ps):
    H = 0
    for P in Ps:
            H -= P * numpy.log2(P)
    return H

def entropy_alt1(Ps):
    PyPs = numpy.array(Ps)
    ent = scipy.stats.entropy(PyPs)
    return ent

def entropy_alt2(Ps):
    PyPs = numpy.array(Ps)
    ent = scipy.special.entr(PyPs)
    return ent

def SumN_list(general_dict, documents):
    SumN_list = []
    for word in general_dict:
        SumN = get_SumN(word,documents)
        SumN_list.append(SumN)
    return SumN_list

def entropy_list(general_dict, documents):
    entropy_list=[]
    for word in general_dict:
        Ns = getNs(word, documents)
        SumN = sum(Ns)
        Ps = P_list(Ns, SumN)
        entropy = entropy_alt1(Ps)
        entropy_list.append(entropy)
    return entropy_list

def max_entropy(corpus):
    n = len(corpus)
    Hmax = numpy.log2(n)
    return Hmax

# pos or neg documents --> list of strings ['sentence is sentence', 'sentence 2']
# pos_entropies = [pe/max_entropy(positive_documents) for pe in entropy_list(general_dict, positive_documents)]
# pos_entropies = [ne/max_entropy(negative_documents) for ne in entropy_list(general_dict, negative_documents)]
def make_pos_neg_keywords_alpha(alpha, pos_entropies, neg_entropies, general_dict):
    pos_dict = []
    neg_dict = []
    maxnum = len(pos_entropies)
    for i in range(maxnum):
        word = general_dict[i]
        Hpj = pos_entropies[i]
        Hnj = neg_entropies[i]
        if (Hpj>(alpha*Hnj)):
            pos_dict.append(word)
        elif (Hnj>(alpha*Hpj)):
            neg_dict.append(word)
    return pos_dict, neg_dict

if __name__ == '__main__':
    pass

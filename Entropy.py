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
def general_dictionary(corpus):
    tokenized = Tokenize(corpus)
    dictionary = Dictionary(tokenized)
    general_dictionary = sorted([word for word in dictionary.values()])
    return general_dictionary

# For a particular word in either category (positive or negative, etc.)
# The following methods should be used in a list of documents of ONLY positive or ONLY negative documents

# Format is list of documents
def getNs(word, documents):
    Ns = []
    for sentence in documents:
        N = sentence.count(word)
        Ns.append(N)
    return N

def SumN(word, documents):
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

def SumN_list(general_dictionary, documents):
    SumN_list = []
    for word in general_dictionary:
        SumN = SumN(word,documents)
        SumN_list.append(SumN)
    return SumN_list

def entropy_list(general_dictionary, documents):
    entropy_list=[]
    for word in general_dictionary:
        Ns = getNs(word, documents)
        SumN = sum(Ns)
        Ps = P_list(Ns, SumN)
        entropy = entropy(Ps)
        entropy_list.append(entropy)
    return entropy_list

def max_entropy(corpus):
    n = len(corpus)
    Hmax = numpy.log2(n)
    return Hmax

def make_pos_neg_keywords_alpha(alpha, pos_entropies, neg_entropies, pos_sums, neg_sums, general_dictionary):
    pos_dict = []
    neg_dict = []
    maxnum = len(pos_entropies)
    for i in xrange(maxnum):
        word = general_dictionary[i]
        Hpj = pos_entropies[i]
        Hnj = neg_entropies[i]
        Spj = pos_sums[i]
        Snj = neg_sums[i]
        if (Hpj>(alpha*Hnj)):
            pos_dict.append(word)
        elif (Hnj>(alpha*Hpj)):
            neg_dict.append(word)
        elif Spj==0 and Snj>0:
            neg_dict.append(word)
        elif Snj==0 and Spj>0:
            pos_dict.append(word)
    return pos_dict, neg_dict

if __name__ == '__main__':
    pass

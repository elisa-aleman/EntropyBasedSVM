#-*- coding: utf-8 -*-

import numpy
import scipy.stats
import scipy.special

# For a particular word in either category (positive or negative, etc.)
# The following methods should be used in a list of documents of ONLY positive or ONLY negative documents

def getNs(word, documents):
    '''
    Format is list of documents
    '''
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

def make_pos_neg_keywords_alpha(alpha, pos_entropies, neg_entropies, general_dict):
    '''
    pos or neg documents --> list of strings ['sentence is sentence', 'sentence 2']
    pos_entropies = [pe/max_entropy(positive_documents) for pe in entropy_list(general_dict, positive_documents)]
    pos_entropies = [ne/max_entropy(negative_documents) for ne in entropy_list(general_dict, negative_documents)]
    '''
    pos_keys = []
    neg_keys = []
    maxnum = len(pos_entropies)
    for i in range(maxnum):
        word = general_dict[i]
        Hpj = pos_entropies[i]
        Hnj = neg_entropies[i]
        if (Hpj>(alpha*Hnj)):
            pos_keys.append(word)
        elif (Hnj>(alpha*Hpj)):
            neg_keys.append(word)
    return pos_keys, neg_keys

if __name__ == '__main__':
    pass

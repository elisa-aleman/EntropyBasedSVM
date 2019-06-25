#-*- coding: utf-8 -*-

import gensim

def LemmatizeEnglish(content):
    lem = ' '.join([i.decode('utf-8').split('/')[0] for i in gensim.utils.lemmatize(content)])
    return lem

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
    
if __name__ == '__main__':
    pass
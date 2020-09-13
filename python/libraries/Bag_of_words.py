#-*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
import numpy

def Vectorize_training_data_Bag_of_Words(training_sentences, word_list):
    '''
    training_sentences: ["text",1 or 0] 1: positive, 0: negative
    word_list = ['word', 'word', 'word']
    '''
    corpus = [sentence[0] for sentence in training_sentences]
    y_list = [sentence[1] for sentence in training_sentences]
    X = Bag_of_Words(corpus, word_list)
    y = numpy.array(y_list)
    return X, y

def Bag_of_Words(corpus, word_list):
    '''
    corpus: ["text1", "text2", ...]
    word_list = ['word', 'word', 'word']
    '''
    vectorizer = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')
    IM = vectorizer.fit_transform(word_list)
    X = vectorizer.transform(corpus).toarray()
    return X

def get_feature_names(keyword_list):
    vectorizer = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')
    IM = vectorizer.fit_transform(keyword_list)
    features = vectorizer.get_feature_names()
    return features

if __name__ == '__main__':
    pass

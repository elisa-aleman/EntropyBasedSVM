#-*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer

def Vectorize_Bag_of_Words(sentences, word_list):
    # sentences: ["text",1 or 0] 1: positive, 0: negative
    # word_list = ['word', 'word', 'word']
    corpus = [sentence[0] for sentence in sentences]
    y_list = [sentence[1] for sentence in sentences]
    X = Bag_of_Words(corpus, word_list)
    y = numpy.array(y_list)
    return X, y

def Bag_of_Words(corpus, word_list):
    # corpus: ["text1", "text2", ...]
    # word_list = ['word', 'word', 'word']
    vectorizer = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')
    IM = vectorizer.fit_transform(word_list)
    X = vectorizer.transform(corpus).toarray()
    return X

def Feature_Names(keyword_list):
    vectorizer = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')
    IM = vectorizer.fit_transform(keyword_list)
    features = vectorizer.get_feature_names()
    return features

if __name__ == '__main__':
    pass

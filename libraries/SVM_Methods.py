#-*- coding: utf-8 -*-

from sklearn import svm
import scipy
import numpy
from libraries.ModelMetrics import *

def Predict(clf, sentence, keyword_list):
    vectorizer = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b') #All words included
    IM = vectorizer.fit_transform(keyword_list)
    pre_vector = vectorizer.transform([sentence]).toarray().tolist()
    vector = numpy.array(pre_vector[0])
    predicted = clf.predict(vector.reshape(1,-1))[0]
    return predicted

def SVM_Train(x, y, test_size, shuffle=True, kerne ='linear', C=1.0, gamma=0.001):
    train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    testsize = len(test_y)
    #Define classifier
    clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
    clf.fit(train_x,train_y)
    #Test data
    y_preds = []
    for i in range(testsize):
        predicted = clf.predict(test_x[i].reshape(1,-1))[0]
        y_preds.append(predicted)
    return clf, test_y, y_preds

def SVM_Kfolds(x, y, k, kernel='linear', C=1.0, gamma=0.001):
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = SVM_Train(x, y, test_size, shuffle=True, kernel=kernel, C=C, gamma=gamma)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    results = F_score_Kfolds(true_ys_list, y_pred_list)
    return clf, results

def SVM_weights_untrained(x, y, feature_names, kernel = 'linear', C = 1.0, gamma = 0.001):
    if type(x) == type([]):
        x = numpy.array(x)
    if type(y) == type([]):
        y = numpy.array(y)
    clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
    clf.fit(x,y)
    weights = clf.coef_.tolist()[0]
    influences = list(zip(feature_names, weights))
    return influences

def SVM_weights_trained(clf,keyword_list):
    weights = clf.coef_.tolist()[0]
    feature_names = Feature_Names(keyword_list)
    influences = list(zip(feature_names, weights))
    return influences

if __name__ == "__main__":
    pass

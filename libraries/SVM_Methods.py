#-*- coding: utf-8 -*-

import scipy
import numpy
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from libraries.Model_metrics import F_score_multiclass_Kfolds, F_score_Kfolds
from libraries.Bag_of_words import get_feature_names

def SVM_Train(x, y, test_size, shuffle=True, kernel ='linear', C=1.0, gamma=0.001):
    if test_size>0:
        train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=test_size, shuffle=shuffle)
    else:
        train_x = x
        train_y = y
        test_x = []
        test_y = []
    testsize = len(test_y)
    #Define classifier
    clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
    clf.fit(train_x,train_y)
    #Test data
    y_preds = []
    if test_size>0:
        for i in range(testsize):
            predicted = clf.predict(test_x[i].reshape(1,-1))[0]
            y_preds.append(predicted)
    return clf, test_y, y_preds

def SVM_Kfolds(x, y, k, kernel='linear', C=1.0, gamma=0.001, multiclass=False, with_counts=True, with_lists=True, with_confusion_matrix=True):
    test_size = len(y)//k
    y_pred_list = []
    true_ys_list = []
    for t in range(k):
        clf, test_y, y_preds = SVM_Train(x, y, test_size, shuffle=True, kernel=kernel, C=C, gamma=gamma)
        y_pred_list.append(y_preds)
        true_ys_list.append(test_y)
    if multiclass:
        results = F_score_multiclass_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists, with_confusion_matrix=with_confusion_matrix)
    else:
        results = F_score_Kfolds(true_ys_list, y_pred_list, with_counts=with_counts, with_lists=with_lists)
    return results

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
    feature_names = get_feature_names(keyword_list)
    influences = list(zip(feature_names, weights))
    return influences

if __name__ == "__main__":
    pass

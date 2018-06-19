#-*- coding: utf-8 -*-

from sklearn import svm
import scipy
from sklearn.feature_extraction.text import CountVectorizer
import numpy 

def Vectorize(sentences, dictionary):
    # sentences: ["text",1 or 0] 1: positive, 0: negative
    vectorizer = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')
    IM = vectorizer.fit_transform(dictionary)
    # Method 1
    X_list = []
    y_list = []
    for i in sentences:
        vector = vectorizer.transform([i[0]]).toarray().tolist()
        X_list.append(vector[0])
        y_list.append(i[1])
    X = numpy.array(X_list)
    y = numpy.array(y_list)
    return X, y

def SVMKFolds(k, sentences, dictionary, kernel = 'linear', C = 1.0, gamma = 0.001, times = 1):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(sentences)//k
    # Correct Prediction, True Positive, True Negative, Incorrect Prediction, False Positive, False Negative
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in range(k*times)]
    for t in range(k*times):
        numpy.random.shuffle(sentences)
        X, y = Vectorize(sentences, dictionary)
        #Define classifier
        clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
        clf.fit(X[:-testsize],y[:-testsize])
        #Test data
        for i in range(1, testsize+1):
            predicted = clf.predict(X[-i].reshape(1,-1))[0]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / ((counts[t]["TP"] + counts[t]["FP"])*1.0)
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / ((counts[t]["TP"] + counts[t]["FN"])*1.0)
        else:
            recall = 0
        accuracy = counts[t]["CP"]/(testsize*1.0)
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/(len(precisions)*1.0)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/(len(recalls)*1.0)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/(len(accuracies)*1.0)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/(len(f1s)*1.0)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results

def SVM_weights(x, y, feature_names, kernel = 'linear', C = 1.0, gamma = 0.001):
    if type(x) == type([]):
        x = numpy.array(x)
    if type(y) == type([]):
        y = numpy.array(y)
    clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
    clf.fit(x,y)
    weights = clf.coef_.tolist()[0]
    influences = zip(feature_names, weights)
    return influences

if __name__ == "__main__":
    pass
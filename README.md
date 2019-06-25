# Entropy-Based-SVM
Entropy based binary SVM classifier library

I use entropy in positive and negative emotional classification via SVM in many projects. These are general methods that can be applied in many circumnstances. 

* __Corpus_preprocessing:__ I use gensim to preprocess the corpus into a lemmatized and tokenized version of the texts.

* __Entropy:__ I use scikit and gensim to calculate the entropy of words in positive and negative documents, so that I can then compare both entropies of the word and know the words that are probabilistically evenly distributed in one category but not in the other, which aids in classification.

* __SVM_Methods:__ The methods I use to analyze SVM training results from K-folds cross validation to weight analysis.

* __Model_Metrics:__ I use scikit-learn to write my own K-folds method that returns F1, Accuracy, Precision and Recall. Included methods usually only return Accuracy or F1.





# Entropy-Based-SVM
Entropy based binary SVM classifier library

I use entropy in positive and negative emotional classification via SVM in many projects. These are general methods that can be applied in many circumnstances. 

* __BagOfConcepts:__ Provided a dictionary of clustered words, it returns a Bag of Concepts vector for a corpus

* __BagOfWords:__ Provided a list of words, it returns a Bag of Words vector for a corpus

* __Corpus_preprocessing:__ I use gensim to preprocess the corpus into a lemmatized and tokenized version of the texts.

* __Entropy:__ I use scikit and gensim to calculate the entropy of words in positive and negative documents, so that I can then compare both entropies of the word and know the words that are probabilistically evenly distributed in one category but not in the other, which aids in classification.

* __Model_Metrics:__ I use scikit-learn to write my own K-folds method that returns F1, Accuracy, Precision and Recall. Included methods usually only return Accuracy or F1.

* __Posi-Nega-Neutra_Tagged-Sentence-Parsing:__ When creating my training data, I xml tagged the text with <positive>, <negative> and <neutral> tags. This method helps me parse that to a python list.

* __SVM_Methods:__ The methods I use to analyze SVM training results from K-folds cross validation to weight analysis.


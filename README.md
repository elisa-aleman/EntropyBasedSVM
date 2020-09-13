# Entropy-Based-SVM
Entropy based binary SVM classifier library

I use entropy in positive and negative emotional classification via SVM in many projects. These are general methods that can be applied in many circumnstances. 

* __Bag_of_concepts:__ Provided a dictionary of clustered words, it returns a Bag of Concepts vector for a corpus

* __Bag_of_words:__ Provided a list of words, it returns a Bag of Words vector for a corpus

* __Corpus_preprocessing:__ I use gensim to preprocess the corpus into a lemmatized and tokenized version of the texts.

* __Entropy:__ I use scikit and gensim to calculate the entropy of words in positive and negative documents, so that I can then compare both entropies of the word and know the words that are probabilistically evenly distributed in one category but not in the other, which aids in classification.

* __Posi-Nega-Neutra_Tagged-Sentence-Parsing:__ When creating my training data, I xml tagged the text with <positive>, <negative> and <neutral> tags. This method helps me parse that to a python list.

* __SVM_Methods:__ The methods I use to analyze SVM training results from K-folds cross validation to weight analysis.

* __Model_methods:__ Not only to use SVM, but when I wish to use other machine learning methodologies.

* __Model_metrics:__ I use scikit-learn to write my own K-folds method that returns F1, Accuracy, Precision and Recall. Included methods usually only return Accuracy or F1.

* __Kaomoji:__ A library to detect kaomoji in text and convert them into numbered tags before applying segmentation by parsers. (in languages without spaces like Chinese) 

* __ProjectPaths:__ Paths to folders inside the project, such as "data", "logs", etc. for organization.

* __UsefulMethods:__ A few methods I use constantly

* __Best_SVM_selection:__ I use this constantly to compare different SVM parameters and choose the best, so I made it more accessible to import

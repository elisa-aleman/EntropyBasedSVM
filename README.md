# Entropy-Based-SVM
Entropy based binary SVM classifier library

I use entropy in positive and negative emotional classification via SVM in many projects. These are general methods that can be applied in many circumnstances. 

* __Entropy:__ I use scikit and gensim to calculate the entropy of words in positive and negative documents, so that I can then compare both entropies of the word and know the words that are probabilistically evenly distributed in one category but not in the other, which aids in classification.

* __SVM K-folds:__ I use scikit-learn to write my own K-folds method that returns F1, Accuracy, Precision and Recall. Included methods usually only return Accuracy or F1.

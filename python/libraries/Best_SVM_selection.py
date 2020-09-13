#-*- coding: utf-8 -*-

from libraries.ProjectPaths import *
from libraries.UsefulMethods import *
from libraries.Entropy import *
from libraries.SVM_methods import *
from libraries.Bag_of_words import *

def make_entropies_table(corpus, positive_documents, negative_documents, general_dict, file_prefix=''):
    pos_raw_entropies = entropy_list(general_dict, positive_documents)
    pos_entropies = [pe/max_entropy(positive_documents) for pe in pos_raw_entropies]
    neg_raw_entropies = entropy_list(general_dict, negative_documents)
    neg_entropies = [ne/max_entropy(negative_documents) for ne in neg_raw_entropies]
    ins_table = []
    # new_titles = ["Word_ID"," Word","Positive_Entropy","Negative_Entropy"]
    for num, word in enumerate(general_dict):
        ins_row = (num+1,word,pos_entropies[num],neg_entropies[num])
        ins_table.append(ins_row)
    ins_table_df = pandas.DataFrame(ins_table,columns=["Word_ID","Word","Positive_Entropy","Negative_Entropy"])
    ins_path = make_data_path("{}entropies.csv".format(file_prefix))
    ins_table_df.to_csv(ins_path, index=False)

def get_entropies(entropies_df):
    pos_entropies = entropies_df.Positive_Entropy.tolist()
    neg_entropies = entropies_df.Negative_Entropy.tolist()
    return pos_entropies, neg_entropies

def Make_Keyword_Lists(alphas, general_dict, entropies_df, file_prefix=''):
    pos_entropies, neg_entropies = get_entropies(entropies_df)
    for alpha in alphas:
        print('Making Keyword List for alpha: {}'.format(alpha))
        pos_dict, neg_dict = make_pos_neg_keywords_alpha(alpha, pos_entropies, neg_entropies, general_dict)
        pos_file = make_keyword_path('{}Positive_Keywords_alpha_{}.txt'.format(file_prefix, alpha))
        neg_file = make_keyword_path('{}Negative_Keywords_alpha_{}.txt'.format(file_prefix, alpha))
        for pos_word in pos_dict: print_log(pos_word,pos_file)
        for neg_word in neg_dict: print_log(neg_word,neg_file)
    print('Done')

def read_keyword_list(alpha, file_prefix=''):
    pos_file = make_keyword_path('{}Positive_Keywords_alpha_{}.txt'.format(file_prefix, alpha))
    neg_file = make_keyword_path('{}Negative_Keywords_alpha_{}.txt'.format(file_prefix, alpha))
    pos_dict = read_dict(pos_file)
    neg_dict = read_dict(neg_file)
    return pos_dict,neg_dict

def SVMResults(training_sentences, alphas, k, Cs, kernel='linear', file_prefix=''):
    # training_sentences: ["text",1 or 0] 1: positive, 0: negative
    svm_results_table = []
    pos_results_table = []
    neg_results_table = []
    # new_titles = ["Keywords","Alpha","C","kernel","Average_F1","StDv_F1","Average_Accuracy","StDv_Accuracy","Average_Precision","StDv_Precision","Average_Recall","StDv_Recall"]
    for alpha in alphas:
        pos_dict, neg_dict = read_keyword_list(alpha=alpha, file_prefix=file_prefix)
        pos_x, pos_y = Vectorize_training_data_Bag_of_Words(training_sentences, pos_dict)
        neg_x, neg_y = Vectorize_training_data_Bag_of_Words(training_sentences, neg_dict)
        for C in Cs:
            print('Calculating SVM Kfold results for alpha: {} and C: {}'.format(alpha,C))
            pos_results = SVM_Kfolds(pos_x, pos_y, k, kernel=kernel, C=C, multiclass=False, with_counts=False, with_lists=False)
            neg_results = SVM_Kfolds(neg_x, neg_y, k, kernel=kernel, C=C, multiclass=False, with_counts=False, with_lists=False)
            pos_row = ("Positive",alpha,C,kernel,pos_results["F1"]["average"],pos_results["F1"]["std"],pos_results["accuracy"]["average"],pos_results["accuracy"]["std"],pos_results["precision"]["average"],pos_results["precision"]["std"],pos_results["recall"]["average"],pos_results["recall"]["std"])
            neg_row = ("Negative",alpha,C,kernel,neg_results["F1"]["average"],neg_results["F1"]["std"],neg_results["accuracy"]["average"],neg_results["accuracy"]["std"],neg_results["precision"]["average"],neg_results["precision"]["std"],neg_results["recall"]["average"],neg_results["recall"]["std"])
            svm_results_table.append(pos_row)
            svm_results_table.append(neg_row)
    svm_results_table_df = pandas.DataFrame(svm_results_table,columns=["Keywords","Alpha","C","kernel","Average_F1","StDv_F1","Average_Accuracy","StDv_Accuracy","Average_Precision","StDv_Precision","Average_Recall","StDv_Recall"])
    svm_results_filename = make_log_file('{}SVM_Results.csv'.format(file_prefix))
    svm_results_table_df.to_csv(svm_results_filename, index=False)
    print('Done')

def SelectBestSVM(file_prefix=''):
    svm_select_log = make_log_file("{}Best_SVMs_select.txt".format(file_prefix))
    svm_results_filename = make_log_file('{}SVM_Results.csv'.format(file_prefix))
    svm_results_df = pandas.read_csv(svm_results_filename)
    positives_df = svm_results_df[svm_results_df.Keywords=="Positive"]
    negatives_df = svm_results_df[svm_results_df.Keywords=="Negative"]
    # take highest Average_F1 for Keywords==Positive and Keywords==Negative
    best_pos = positives_df[positives_df['Average_F1']==positives_df['Average_F1'].max()]
    print_STD_log('Best Performing Positive SVM',svm_select_log)
    print_STD_log('Alpha: {}, C: {}'.format(best_pos.Alpha.values[0], best_pos.C.values[0]),svm_select_log)
    print_STD_log('F1: {} F1_stdv: {}'.format(best_pos.Average_F1.values[0], best_pos.StDv_F1.values[0]),svm_select_log)
    print_STD_log('Acc: {} Acc_stdv:{}'.format(best_pos.Average_Accuracy.values[0], best_pos.StDv_Accuracy.values[0]),svm_select_log)
    print_STD_log('##########',svm_select_log)
    best_neg = negatives_df[negatives_df['Average_F1']==negatives_df['Average_F1'].max()]
    print_STD_log('Best Performing Negative SVM',svm_select_log)
    print_STD_log('Alpha: {}, C: {}'.format(best_neg.Alpha.values[0], best_neg.C.values[0]),svm_select_log)
    print_STD_log('F1: {} F1_stdv: {}'.format(best_neg.Average_F1.values[0], best_neg.StDv_F1.values[0]),svm_select_log)
    print_STD_log('Acc: {} Acc_stdv:{}'.format(best_neg.Average_Accuracy.values[0], best_neg.StDv_Accuracy.values[0]),svm_select_log)
    print_STD_log('##########',svm_select_log)


if __name__ == '__main__':
    pass
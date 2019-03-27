"""
Feature Engineering with Text, Date, Categorical, and Numerical Features to Predict Faulty Loans!
DataRobot Challenge
*************************************************************
**Author**: `Isabel Metzger <https://github.com/izzykayu/DefaultLoanPrediction>`_
 Contact: im1247@nyu.edu

In this project I create a logistic regression model to identify loan

Then, we will be teaching a neural network to identify loan defaults using the same text features
 except rather than one-hot encoding of preprocessed words, text is initialized on glove embeddings



**Requirements**
nltk
sklearn
textblob
keras
tensorflow
afinn
vadersentiment
pandas
matplotlib
numpy

"""
# importing various packages
import warnings; warnings.filterwarnings('ignore')
import collections
from random import randrange
from nltk import sent_tokenize, word_tokenize
import json
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.stem import PorterStemmer
import string
import unicodedata
import matplotlib.pyplot as plt
import operator
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as sklm
from sklearn.model_selection import train_test_split
import os
import numpy as np

import random
from nltk.corpus import stopwords
import re
from textblob import TextBlob
import pickle
import pprint

from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

afinn_sentiment_analyzer = Afinn(emoticons=False)
vader_sentiment_analyzer = SentimentIntensityAnalyzer()

import pandas as pd

stopwords = set(stopwords.words("english"))
pd.set_option('display.max_columns', 50)
random.seed(28)

######################################################################
######################################################################
# DATA PRE-PROCESSING AND SPLITTING SETS
# --------------------------
# 1. Partition your data into a holdout set and 5 stratified CV folds.
#
# The functions below are to create the 5 fold cross-validation set and hold-out set.
# Other functions are utility functions such as create directory and etc.
# I also create a validation set because I always like to use this to determine which model I will pick.
# After I have chosen which model I believe I would employ, I will test it on the hold-out test set to see how my assumptions were
#

def create_directory(directory):
    """
    input is directory name/path
    output of this fxn creates directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_results(pred, y_true, path):

    create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (x, y) in zip( pred, y_true):
            f.write("{:.6f},{}\n".format(x, y))


def make_initial_training_test_set(dataframe, column_names_X):
    """
    this function makes train and test split
    :param dataframe: pandas dataframe
    :param column_names_X: list of variables to be included in the split
    :return:X variables of first and second splits, y target variables for first and second splits
    and indices (Ids) for these splits
    """
    dataframe_X = dataframe[column_names_X].values
    indices = dataframe['Id'].values
    target = dataframe['is_bad'].values
    X_train, tosplit_X, y_train, tosplit_y, idx1, idx2 = train_test_split(dataframe_X, target, indices, stratify=target,
                                                                          random_state=28, test_size=.30)
    print('shape of first set:', X_train.shape)
    print('shape of second set:', tosplit_X.shape)
    return X_train, tosplit_X, y_train, tosplit_y, idx1, idx2


df = pd.read_csv('../data/loanpred/DR_Demo_Lending_Club.csv')
print("**Loaded full dataset**\n")
print(len(df.columns), 'columns including the id column and prediction variable column:\n', list(df.columns))
dtypesdf = df.columns.to_series().groupby(df.dtypes).groups
print('\n**DataSet variable types**\n')
pprint.pprint(dtypesdf)

# Store data (information about columns)
with open('../data/dtypes_original_df_json.pickle', 'wb') as handle:
    pickle.dump(dtypesdf, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_ground_truths_train_test(y_train, y_val, y_test, idx1, idx2, idx3):
    """
    this function writes out three csv files in data of the ground truths and the corresponding Ids
    :param y_train: ground truths of y_train
    :param y_val: " " of y_val
    :param y_test: "" of y_test
    :param idx1: indices y_train
    :param idx2: " y_val
    :param idx3: "y_test
    :return: csv files are written out., information about the number of observations included
    """
    train_gt = pd.DataFrame(idx1)
    train_gt[1] = y_train
    val_gt = pd.DataFrame(idx2)
    val_gt[1] = y_val
    test_gt = pd.DataFrame(idx3)
    test_gt[1] = y_test
    print('\n***********\n')
    train_gt.to_csv('../data/gt_train.csv', index=False, header=False)
    print('training set # of observations:', train_gt.shape[0])
    val_gt.to_csv('../data/gt_val.csv', index=False, header=False)
    print('val set # of observations:', val_gt.shape[0])
    test_gt.to_csv('../data/gt_test.csv', index=False, header=False)
    print('test set # of observations:', test_gt.shape[0])
    print('***********wrote out training set, val (dev) set, and test set ground truths')


X_train, X_tosplitCV, y_train, y_tosplitCV, train_indices, tosplit_indices = make_initial_training_test_set(df, column_names_X = ['emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'pymnt_plan', 'Notes', 'purpose_cat', 'purpose', 'zip_code', 'addr_state', 'debt_to_income', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code'])
nontrain_df = df[~df.Id.isin(list(set(train_indices)))]
X_val, X_test, y_val, y_test, val_indices, test_indices = make_initial_training_test_set(nontrain_df, column_names_X = ['emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'pymnt_plan', 'Notes', 'purpose_cat', 'purpose', 'zip_code', 'addr_state', 'debt_to_income', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code'])

create_ground_truths_train_test(y_train, y_val, y_test, train_indices, val_indices, test_indices)
print("\n\n**TRAIN, VALIDATION, and TEST SET CREATION**\n")

train_df = df[df.Id.isin(train_indices)] # 7000 observations


def cv_split(dataset, folds=5):
    """
    function inspired by https://machinelearningmastery.com/implement-resampling-methods-scratch-python/
    :param dataset: pandas dataframe such as train_df
    :param folds: 5 different sets
    :return: returns dataset split which is a list of lists containing the ids for 5 cross fold
    """
    dataset = dataset.Id.values
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

list_of_datasplit_ids = cv_split(train_df, 5)

def write_gt(id_list, ground_truth_df, csvname):
    csv_to_write = ground_truth_df[ground_truth_df.Id.isin(id_list)]
    csv_to_write.to_csv(os.path.join('../data', csvname), header=False, index=False)
    print('wrote out csv {} with shape {}'.format(csvname,csv_to_write.shape))


def create_training_cv_fols(train_df, id_list):
    number_folds = len(id_list)
    ground_truth_df = train_df[['Id', 'is_bad']]
    for ix in range(number_folds):
        csvname = 'gt_TRAIN_CV_SPLIT' + str(ix) + '.csv'
        list_now = list_of_datasplit_ids[ix]
        write_gt(list_now, ground_truth_df, csvname)
    print('all done creating cross validation ground truths and id splits')


print(create_training_cv_fols(train_df, list_of_datasplit_ids))

val_df = df[df.Id.isin(val_indices)]
test_df = df[df.Id.isin(test_indices)]

print("**TRAIN, VALIDATION, and TEST SET CREATED**\n")
print("example Notes from training set to help with text-preprocessing functions")
notes_train_df = train_df[~train_df.Notes.isna()]
# pprint.pprint(notes_train_df.Notes.values)

# Load previously saved data (deserialize)
with open('../data/dtypes_original_df_json.pickle', 'rb') as handle:
    dtypesdf = pickle.load(handle)

for k, v in dtypesdf.items():
    print('columns w/ dtype ** {} ** include:\n {}'.format(k, list(v)))

######################################################################
# Exploring the unbalanced target dataset
# --------------------------
# Using train set only to avoid leaky features
# we look at class weights and can use this dictionary in the neural net and logregmodels

print('Percentage of target prediction class in training set: {}%'.format(round(100*train_df['is_bad'].sum()/train_df.shape[0],2)))

from sklearn.utils.class_weight import compute_class_weight
y_integers = train_df['is_bad'].values
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))
print('class weights:\n',d_class_weights, '\n****************\n')

######################################################################
######################################################################
# MODEL CREATION
# --------------------------
# 2. Pick any two machine learning algorithms from the list below, and build a binary
# classification model with each of them:
# ○ Regularized Logistic Regression (scikit-learn)
# ○ Gradient Boosting Machine (scikit-learn, XGBoost or LightGBM)
# ○ Neural Network (Keras), with the architecture of your choice
# The models I pick include Logistic Regression and a neural net built in keras

def print_metrics_binary(y_true, predictions, verbose=1):
    """
    prints metrics for loan defaults
    :param y_true: ground truths
    :param predictions: predictions from model
    """
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = sklm.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = sklm.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = sklm.precision_recall_curve(y_true, predictions[:, 1])
    f1_scores = sklm.f1_score(y_true, predictions[:, 1])
    auprc = sklm.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return {
        "acc": acc,
        "prec0": prec0,
        "prec1": prec1,
        "rec0": rec0,
        "rec1": rec1,
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse,
        "f1scores": f1_scores
    }
######################################################################
# LOGISTIC REGRESSION MODEL CREATION BASELINE
# --------------------------
# This baseline model takes in features and uses mean imputation and standard transformer of various inputs

def LogisticRegressionModel(train_X, val_X, test_X, train_y, val_y, test_y):
    print('  train data shape = {}'.format(train_X.shape))
    print('  validation data shape = {}'.format(val_X.shape))
    print('  test data shape = {}'.format(test_X.shape))

    print('Imputing missing values ...')
    imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    print('Normalizing the data to have zero mean and unit variance ...')
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    penalty = 'l2'
    file_name = '{}-C{}-logisticregression-'.format(penalty, 1)

    logreg = LogisticRegression(penalty=penalty, C=1.0, random_state=42, class_weight =d_class_weights)
    logreg.fit(train_X, train_y)

    result_dir = os.path.join('../data', 'results')
    create_directory(result_dir)

    with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
        ret = {k : float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    prediction = logreg.predict_proba(test_X)[:, 1]

    with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(test_y, prediction)
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    save_results(prediction, test_y, os.path.join(result_dir, 'predictions', file_name + '.csv'))
#
# import keras.backend as K
# os.environ['OMP_NUM_THREADS'] = '4'
# from keras.models import Model
# from keras.layers import Dense, Embedding, Input, Concatenate, Conv1D
# from keras.layers import GlobalMaxPool1D, Dropout, SpatialDropout1D
# from keras.preprocessing import text, sequence
#
# from keras.preprocessing.text import Tokenizer
#
# def precision(y_true, y_pred):
#     """Precision metric.
#     Only computes a batch-wise average of precision.
#     Computes the precision, a metric for multi-label classification of
#     how many selected items are relevant.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
#
# def recall(y_true, y_pred):
#     """Recall metric.
#     Only computes a batch-wise average of recall.
#     Computes the recall, a metric for multi-label classification of
#     how many relevant items are selected.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
#
# def fbeta_score(y_true, y_pred, beta=1):
#     """Computes the F score.
#     The F score is the weighted harmonic mean of precision and recall.
#     Here it is only computed as a batch-wise average, not globally.
#     This is useful for multi-label classification, where input samples can be
#     classified as sets of labels. By only using accuracy (precision) a model
#     would achieve a perfect score by simply assigning every class to every
#     input. In order to avoid this, a metric should penalize incorrect class
#     assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
#     computes this, as a weighted mean of the proportion of correct class
#     assignments vs. the proportion of incorrect class assignments.
#     With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
#     correct classes becomes more important, and with beta > 1 the metric is
#     instead weighted towards penalizing incorrect class assignments.
#     """
#     if beta < 0:
#         raise ValueError('The lowest choosable beta is zero (only precision).')
#
#     # If there are no true positives, fix the F score at 0 like sklearn.
#     if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
#         return 0
#
#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     bb = beta ** 2
#     fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
#     return fbeta_score
#
#
# def fmeasure(y_true, y_pred):
#     """Computes the f-measure, the harmonic mean of precision and recall.
#     Here it is only computed as a batch-wise average, not globally.
#     """
#     return fbeta_score(y_true, y_pred, beta=1)
#
# def make_neural_net(train, test, non_word_input_columns):
#     y = train['is_bad'].values
#     max_features = 15500  #
#     maxlen = 7000  # padding length
#     num_folds = 5  # 5 #number of folds
#     list_sentences_test = test["clean_text"].fillna("").values
#
#     print('mean text len:', train["clean_text"].str.count('\S+').mean())
#     print('max text len:', train["clean_text"].str.count('\S+').max())
#
#     # tokenizer = Tokenizer()
#     # tokenizer.fit_on_texts(list(list_sentences_train)) #  + list(list_sentences_test)
#     # num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])
#     # print('num_words',num_words)
#     # max_features = num_words
#     tokenizer = Tokenizer(num_words=max_features)
#     tokenizer.fit_on_texts(list(list_sentences_train))  # + list(list_sentences_test)
#     list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
#     list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
#     print('padding sequences')
#     X_train = {}
#     X_test = {}
#     X_train['text'] = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post', truncating='post')
#     X_test['text'] = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post', truncating='post')
#     scaler = MinMaxScaler()
#
#     X_train['num_vars'] = scaler.fit_transform(
#         train[[non_word_input_columns]])
#     X_test[non_word_input_columns] = scaler.transform(
#         test[['num_words', 'num_comas', 'num_bangs', 'num_quotas', 'avg_word', 'sentiment']])
#
#     all_embs = np.stack(embeddings_index.values())
#     emb_mean, emb_std = all_embs.mean(), all_embs.std()
#
#     print('create embedding matrix')
#     word_index = tokenizer.word_index
#     nb_words = min(max_features, len(word_index))
#     embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#     for word, i in word_index.items():
#         if i >= max_features:
#             continue
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#
#     def get_model_cnn(X_train):
#         global embed_size
#         inp = Input(shape=(maxlen,), name="text")
#         num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
#         x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#         x = SpatialDropout1D(0.2)(x)
#         z = GlobalMaxPool1D()(x)
#         x = GlobalMaxPool1D()(Conv1D(embed_size, 4, activation="relu")(x))
#         x = Concatenate()([x, z, num_vars])
#         x = Dropout(0.3)(x)
#         x = Dense(13, activation="sigmoid")(x)
#         model = Model(inputs=[inp, num_vars], outputs=x)
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, fmeasure])
#         return model
#
#     print("GLOVE KERAS SIMPLE MODEL! WITH FE! 300Dimension Glovee\n")
#     print('start modeling')
#
#     scores = []
#     predict = np.zeros((test.shape[0], 13))
#     prediction_class = np.zeros((test.shape[0], 13))
#     oof_predict = np.zeros((train.shape[0], 13))
#     f1_scores = []
#     acc_scores = []
#     kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)
#     for train_index, test_index in kf.split(X_train['num_vars']):
#         kfold_X_train = {}
#         kfold_X_valid = {}
#         y_train, y_test = y[train_index], y[test_index]
#         for c in ['text', 'num_vars']:
#             kfold_X_train[c] = X_train[c][train_index]
#             kfold_X_valid[c] = X_train[c][test_index]
#
#         model = get_model_cnn(X_train)
#         model.fit(kfold_X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
#         predict += model.predict(X_test, batch_size=1000) / num_folds
#         # model.predict_generator()
#         # predict_classes += predict.argmax(axis=-1)
#         oof_predict[test_index] = model.predict(kfold_X_valid, batch_size=1000)
#         metrics.average_precision_score(y_test, oof_predict[test_index])
#         print(model.metrics_names)  # , model.metrics_tensors)
#         # scores.append(cv_score)
#         # f1_scores.append(fscore)
#         # print('cv score: ', cv_score)
#         # print('f1 score: ',fscore)
#         # print(classification_report(y_test, oof_predict[test_index]))
#
#     # print('Total CV score is {}'.format(np.mean(scores)))
#     # print('Mean f1 score is {}'.format(np.mean(f1_scores)))
#     sample_submission_classes = pd.DataFrame.from_dict({'ID': test['ID']})
#     sample_submission = pd.DataFrame.from_dict({'ID': test['ID']})
#     oof = pd.DataFrame.from_dict({'ID': train['ID']})
#     for c in list_classes:
#         oof[c] = np.zeros(len(train))
#         sample_submission[c] = np.zeros(len(test))
#     history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
#                         callbacks=[reduce_lr, csv_logger, early, k_metrics], verbose=2, class_weight={0: 1, 1: 90})
#     # val_weights={0:1, 1:90}
#     from keras.models import save_model
#     save_model(model, '/gpfs/home/metzgi01/model.' + str(today) + '_' + embed_name + '_GRU_POOL.h5py')
#     del model
#     from keras.models import load_model
#     load_model('/gpfs/home/metzgi01/model.' + str(today) + '_' + embed_name + '_GRU_POOL.h5py', model)
#     model.predict(x_test, batch_size=1024)
#     print(history.history.keys())
#     print(pd.DataFrame(k_metrics))
#     print('confusion\n', k_metrics.confusion, 'precision\n', k_metrics.precision, 'recall\n',
#           k_metrics.recall, 'f1\n', k_metrics.f1s, 'kappa\n', k_metrics.kappa, 'auc\n', k_metrics.auc)
#     print("\nmodel predicting.. with batch size 1024 ( large batch size )")
#     predictions = model.predict(x_test, batch_size=1024)
#     print(predictions)
#     print(type(predictions))
#     predictions_binary = [round(value) for value in predictions]
#     print("predictions binary\n", predictions_binary)
#     print(type(predictions_binary))
#     prediction_classes = np.round(np.asarray(model.predict(x_test, batch_size=1024)))
#     print(prediction_classes)
#     print("type predictions:", type(prediction_classes))
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('/gpfs/home/metzgi01/pngs/gru_acc_' + str(batch_size) + '_' + str(today) + '_' + embed_name + '.png')
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('/gpfs/home/metzgi01/pngs/gru_loss_' + str(batch_size) + '_' + str(today) + '_' + embed_name + '.png')
#     print(sklm.classification_report(prediction_classes))
#     print("trying to reshape predictions to fit for a dataframe", predictions.reshape(-1, 1))
#     print("done!")
# fscore = f1score = fmeasure
# # making weighted binary cross entropy fxn
# def weighted_binary_crossentropy(y_true, y_pred, weight=1.):
#     y_true = K.clip(y_true, K.epsilon(), 1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
#     logloss = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
#     return K.mean(logloss, axis=-1)
#
# def get_model_cnn(X_train, maxlen):
#     global embed_size
#     inp = Input(shape=(maxlen, ), name="text")
#     num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
#     x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#     x = SpatialDropout1D(0.2)(x)
#     z = GlobalMaxPool1D()(x)
#     x = GlobalMaxPool1D()(Conv1D(embed_size, 4, activation="relu")(x))
#     x = Concatenate()([x,z,num_vars])
#     x = Dropout(0.3)(x)
#     x = Dense(13, activation="sigmoid")(x)
#     model = Model(inputs=[inp,num_vars], outputs=x)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, fmeasure])
#     return model
#
# print("GLOVE KERAS SIMPLE MODEL! WITH FE! 300Dimension Glovee\n")
# print('start modeling')

#######################################################################
# FREE-TEXT DATA feature engineering, pre-processing, and exploration
# --------------------------
#
# The functions below are to create the 5 fold cross-validation set and hold-out set.
# I also create a validation set because I always like to use this to determine which model I will pick.
# After I have chosen which model I believe I would employ, I will test it on the hold-out test set to see how my assumptions were
#
# The functions below are to create the 5 fold cross-validation set and hold-out set.
# I also create a validation set because I always like to use this to determine which model I will pick.
# After I have chosen which model I believe I would employ, I will test it on the hold-out test set to see how my assumptions were
#

######################################################################
# FREE-TEXT DATA feature engineering, pre-processing, and exploration
# --------------------------
#
# The functions below are to create the 5 fold cross-validation set and hold-out set.
# I also create a validation set because I always like to use this to determine which model I will pick.
# After I have chosen which model I believe I would employ, I will test it on the hold-out test set to see how my assumptions were
#

free_text_vars = ['emp_title', 'Notes', 'purpose']
# these are examples randomly selected from the free text variables emp_title, Notes, and purpose

example_notes = ["  Borrower added on 04/14/11 > I will be using this loan to pay off expenses accrued in the last \
six months on my credit cards, due to a combination of job transition, relocation for the job, and medical expenses \
from a broken tibula. I generally overpay my monthly minimum on my debts, so I expect that this loan will be repaid \
sooner than 5 years. I have a steady job working in the information technology field, I've been employed full-time \
in this field for over eight years, and have been with my present employer for seven months in good standing. \
My monthly budget breakdown is 1/3 of my paycheck going to rent and bills, 1/3 going to living and job transit expenses, \
and 1/3  remaining for general spending and payments.<br/>",
'  Borrower added on 05/18/10 > mick credit card consolidation loan - 100% payoff of credit card debt \
- amex, sears, macys and bank of america<br/>',
'  Borrower added on 11/29/11 > Loan is for debt consolidation and will be paid timely. \
Employed in the healthcare industry for 6 years since moving to NV 7 years ago and have always had stable job positions. \
 Thank you very much for your assistance.<br>',
'  Borrower added on 12/14/11 > looking to be debt free in 3 yrs or less!!<br>',
'I live in a family owned home.  It is my parents, but I am allowed to live here as long as I want as long as I pay \
for the taxes and any home improvements the home needs.  \
I am looking for a loan to add a bathroom on the second floor and finish other small home improvements the house currently needs. \
 I have lived here for 5 years and have done many updates already. \
 This is the first major renovation I am doing on the house. I need the loan to get what I cannot do myself done \
  the right way.  I have excellent credit and pride myself on that.',
'  Borrower added on 08/30/11 > credit cards consolidation and doctors bills..<br/>']

example_purposes =  ['DORCAS LOAN','Promote music Album',
 'paythebillsoff','Debt Consolidation and Major Purchase',
 'Bank of America Shysters','I am looking for a better rate.',
 'freddys loan', 'Credit cards to refinance, thanks LC!',
 'Assist in buying used car', "Mike's loan", 'rags to riches ']

example_emp_titles = [ 'NOVA', 'pearl harbor naval shipyard', 'Columbus Community Hospital',
 'Clayton Eye Center','MaCann Engineering', 'CIGNA', 'ConocoPhillips','Netflix, Inc']

print('\n**********************\n',
    '{} samples of Notes randomly selected from training set\n'.format(len(example_notes)),
    '{} samples of purposes randomly selected from training set\n'.format(len(example_purposes)),
    '{} samples of employer title randomly selected from training set\n'.format(len(example_emp_titles)),
    '\n**********************\n')

def unicodeToAscii(s):
    """
    this provides a unicode to ascii fxn (from class)
    :param s: string
    :return: string that has been normalized
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        )

def replaceContractions(s):
    contractions = ["don't","wouldn't","couldn't","shouldn't", "weren't", "hadn't" , "wasn't", "didn't" , "doesn't","haven't" , "isn't","hasn't"]
    for c in contractions:
        s = s.replace( c, c[:-3] +' not')
    return s

def normalizeString(s):
    s = unicodeToAscii(s).lower()
    s = replaceContractions(s)
    s = re.sub('<br>', ' ', s)
    s = re.sub('<br/>',  ' ', s)
    s = re.sub(r"([,;.:!?])", r" \1 ", s)
    s = re.sub('\n', ' ', s)
    s = re.sub('\t', ' ', s)
    s = re.sub('\d', 'd', s)
    s = re.sub(' +', ' ', s)
    s = s.lower().strip()
    return s


def preProc(text):
    """
    text is a string, for example: "Please keep humira refrigeraterd.
    """
    text2 = normalizeString(text)
    tokens = [word for sent in sent_tokenize(text2) for word in
          word_tokenize(sent)]
    tokens = [word.lower() for word in tokens]
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [word for word in tokens if len(word) >= 3]
    stemmer = PorterStemmer()
    try:
        tokens = [stemmer.stem(word) for word in tokens]

    except:
        tokens = tokens

    tagged_corpus = pos_tag(tokens)

    Noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    Verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    lemmatizer = WordNetLemmatizer()


    def pratLemmatiz(token, tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token, 'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token, 'v')
        else:
            return lemmatizer.lemmatize(token, 'n')


    pre_proc_text = " ".join([pratLemmatiz(token, tag) for token, tag in tagged_corpus])

    return pre_proc_text



def print_preprocced_texts(example_notes):
    # takes a list of example notes or list of strings and prints out the preprocessed version
    counter = 0
    for sent in example_notes:
        counter += 1
        print('{}-----{}-----{}'.format(str(counter), normalizeString(sent), preProc(sent)))


print('\n*********\nexample of preprocessed Notes from train set')
print_preprocced_texts(example_notes)
print('\n*********\nexample of preprocessed purposes from train set')
print_preprocced_texts(example_purposes)
print('\n*********\nexample of preprocessed employer titles from train set')
print_preprocced_texts(example_emp_titles)


print('\n\n***MISSING TEXT AND CATEGORICAL VARIABLE TYPES IN TRAIN SET***\n:', train_df[['emp_title', 'emp_length', 'home_ownership', 'verification_status', 'pymnt_plan',
          'Notes', 'purpose_cat', 'purpose', 'zip_code', 'addr_state', 'earliest_cr_line',
          'initial_list_status', 'policy_code']].isnull().sum())

## we can see here that only free text variables such as emp_title, Notes, and purpose have nan values in the training set
# on the other hand earliest_cr_line, a date obj has 4 missing values-- if there are other non-missing data it may be useful to see if we can infer naything with this


def lexical_diversity(my_text_data):
    """
    input is list of text data
    output gives diversity_score
    """
    word_count = len(my_text_data)
    vocab_size = len(set(my_text_data))
    diversity_score = word_count / vocab_size
    return diversity_score

def create_clean_texts(df):
    df['Notes'] = df['Notes'].fillna('m')
    df['clean_text'] = df['Notes'].apply(lambda u: preProc(u))
    return df


def simpletextfeatureengineering(df, colnames):

    for x in colnames:
        df[x] = df[x].fillna("m")
        print(x)
        new_name = 'count_word_raw_' + x
        df[new_name] = df[x].apply(lambda x: len(str(x).split()))
    # Unique word count
        count_unique_word_raw_name = 'count_unique_word_raw_' + x
        df[count_unique_word_raw_name] = df[x].apply(lambda x: len(set(str(x).split())))
    # # Letter count
        df['count_letters_raw_' + x] = df[x].apply(lambda x: len(str(x)))
    # punctuation count
        df["count_punctuations_raw_" + x] = df[x].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    # upper case words count
        df["count_words_upper_raw_" + x] = df[x].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    # title case words count
        df["count_words_title_raw_" + x] = df[x].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    # Number of stopwords
        df["count_stopwords_raw_" + x] = df[x].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))
    # Average length of the words
        df['mean_word_len_raw_' + x] = df[x].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    # Word count percent in each comment:
        df['word_unique_percent_raw_' + x] = df['count_unique_word_raw_' + x] * 100 / df['count_word_raw_' + x]
    # percentage of punctuation
        df['punctuation_percent_raw_' + x] = df['count_punctuations_raw_' + x] * 100 / df['count_word_raw_' + x]
    # lexical diversity
        df['lexical_diversity_'+ x] = df[x].apply(
        lambda x1: lexical_diversity(x1))

    return df

#
# part of speech dictionary
pos_dic = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}
#
# function to check and get the part of speech tag count of a words in a given sentence
def pos_check(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_dic[flag]:
                cnt += 1
    except:
        pass
    return cnt
#
#
def makePOSfeat(df):
    """
    part of speech tagging counts as engineered features
    :param df: pandas data-frame
    :param colnames: column(s) with text to count POS to create features
    :return: data-frame with more engineered features
    """
    x = 'Notes'
    df['Notes'] = df['Notes'].fillna("m")
  #  df['clean_text']
        # part-of-speech tagging for
    df['noun_count' + x] = df[x].apply(lambda x: pos_check(x, 'noun'))
    df['verb_count' + x] = df[x].apply(lambda x: pos_check(x, 'verb'))
    df['adj_count' + x] = df[x].apply(lambda x: pos_check(x, 'adj'))
    df['adv_count' + x] = df[x].apply(lambda x: pos_check(x, 'adv'))
    df['pron_count' + x] = df[x].apply(lambda x: pos_check(x, 'pron'))

    return df

def make_text_features_all(df, colnames1):
    df = create_clean_texts(df)
    df = simpletextfeatureengineering(df, colnames1)
    df = makePOSfeat(df)
    print('\n**original features plus new text engineered features**\n', list(df.columns))
    return df

# train = make_text_features_all(train_df, free_text_vars)
# val = make_text_features_all(val_df, free_text_vars)
# test = make_text_features_all(test_df, free_text_vars)



cat_vars = ['emp_length', 'home_ownership', 'verification_status', 'pymnt_plan', 'initial_list_status',
            'policy_code','addr_state','purpose_cat']
#
other_cat_vars = ['zip_code', 'earliest_cr_line']

num_vars = ['revol_bal', 'mths_since_last_major_derog','annual_inc', 'debt_to_income', 'delinq_2yrs',
          'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec',
          'revol_util', 'total_acc', 'collections_12_mths_ex_med']

######################################################################
# Exploratory Data Analysis
# ----------------
#
# Dealing with categorical and numerical data to see which features need to be consolidated, utilized, imputed, and etc
#

def print_exploratory_data_analysis(train_df, cat_vars, num_vars):
    print('\n**missing data - a look into no-numerical data**\n')
    print(train_df[cat_vars].isnull().sum())
    for var in cat_vars:
        my_dict = pd.DataFrame(train_df[var].value_counts()).to_dict()

        d = my_dict[var]
        sorted_x = sorted(d.items(), key=lambda kv: kv[1])
        WantedOutput = collections.OrderedDict(sorted_x)
        WantedOutput = dict(WantedOutput)
        print('\n{}** unique categories: {}\t{}'.format(var, train_df[var].nunique(), WantedOutput))
        # sorted(zip(d.values(), d.keys()))
    print('\n**missing data - a look into no-numerical data**')
    print(train_df[num_vars].isnull().sum())
    for x in num_vars:
        print('\n{}\n{}'.format(x, train_df[x].describe()))


print_exploratory_data_analysis(train_df, cat_vars, num_vars)

#
######################################################################
# FEATURE ENGINEERING - PREPROCESSING DATA
# ----------------
#
# From the training set, we will create a categorical transformation map choosing the simplest first
# because payment plan PYMNT_PLAN** unique categories: 2	{'y': 2, 'n': 6998} and
# INITIAL_LIST_STATUS** unique categories: 2	{'m': 13, 'f': 6987}
# provide extremely unbalanced proportions, with over 95% of the data leaning to have only one category, we will remove these variables
# ADDR_STATE** unique categories: 50	{'IA': 1, 'ME': 1, 'ID': 1, 'NE': 2, 'IN': 2, 'TN': 4, 'MS': 7, 'SD': 13, 'VT': 14, 'WY': 17, 'HI': 18, 'MT': 19, 'AK': 21, 'DE': 26, 'NM': 32, 'NH': 32, 'WV': 33, 'AR': 34, 'RI': 37, 'KS': 41, 'UT': 42, 'DC': 42, 'KY': 67, 'OK': 67, 'LA': 71, 'NV': 78, 'OR': 81, 'AL': 84, 'WI': 84, 'MN': 93, 'SC': 93, 'MO': 112, 'CO': 117, 'MI': 130, 'NC': 137, 'CT': 141, 'AZ': 153, 'WA': 161, 'OH': 192, 'MD': 199, 'MA': 230, 'IL': 255, 'PA': 258, 'GA': 264, 'VA': 275, 'NJ': 324, 'TX': 509, 'FL': 514, 'NY': 677, 'CA': 1195}
# State is a great feature to use, however, we would need to minimize the categories such as into representations due to the large bias of states included in this dataset
# I would use the american census data to map state to region and use region instead (thoughts for future work)
# POLICY_CODE** unique categories: 5{'PC2': 1364, 'PC4': 1381, 'PC1': 1397, 'PC5': 1406, 'PC3': 1452} however makes sense to utilize

pc_map = {'PC2': 2, 'PC4': 4, 'PC1': 1, 'PC5': 5, 'PC3': 3,'OTHER': 6, '': 0}
g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}
ve_map = {'VERIFIED - income source': 1, 'VERIFIED - income': 1, 'not verified': 2, 'OTHER': 3, '': 0}

def transform_policy_code(policy_code_series):
    global pc_map
    return {'policy_code': policy_code_series.fillna('').apply(lambda s: pc_map[s] if s in pc_map else pc_map['OTHER'])}

def transform_verification_status(series):
    global ve_map
    return {'verification_status': series.fillna('').apply(lambda s: ve_map[s] if s in ve_map else ve_map['OTHER'])}

ho_map = {'NONE': 0, 'OTHER': 4, 'OWN': 1, 'MORTGAGE': 2, 'RENT': 3}


def transform_home_ownership(series):
    global ho_map
    return {'home_ownership': series.fillna('').apply(lambda s: ho_map[s] if s in ho_map else ho_map['OTHER'])}
# EMP_LENGTH** unique categories: 14	{'33': 1, '11': 2, '22': 4, 'na': 173, '8': 238, '9': 242, '7': 298, '6': 384, '5': 542, '4': 600, '3': 717, '2': 825, '1': 1485, '10': 1489}

e_map = {

    'UNKNOWN': 0,
    '': 0
}
cat_vars_wanted = ['emp_length', 'home_ownership', 'verification_status','policy_code','purpose_cat']

def transform_categorical_features(train_df):
    df_new = {# 'emp_length': [],
              'home_ownership': [], 'verification_status': [], 'policy_code': []}
             #'purpose_cat'
    # }
    train_df1 = train_df
    train_df1['home_ownership']  = pd.Series(transform_home_ownership(train_df['home_ownership'])['home_ownership'])
   # train_df1['emp_length']   = pd.Series(transform_home_ownership(train_df['home_ownership'])['home_ownership'])
    train_df1['policy_code'] = pd.Series(transform_policy_code(train_df['policy_code'])['policy_code'])
    train_df1['verification_status'] = pd.Series(transform_verification_status(train_df['verification_status'])['verification_status'])
    train_df1 = make_text_features_all(train_df1, free_text_vars)
    # print(set(train_df1.columns))
    return train_df1

tdf = transform_categorical_features(train_df.head(20))
tdf.to_csv('example_cat.csv', index=False)
dtypesdf = tdf.columns.to_series().groupby(tdf.dtypes).groups
print('\n**DataSet variable types**\n')
pprint.pprint(dtypesdf)

col_kept_list = ['Id', 'is_bad', 'home_ownership', 'verification_status', 'revol_bal',
       'mths_since_last_major_derog', 'policy_code',
       'count_word_raw_emp_title', 'count_unique_word_raw_emp_title',
       'count_letters_raw_emp_title', 'count_punctuations_raw_emp_title',
       'count_words_upper_raw_emp_title', 'count_words_title_raw_emp_title',
       'count_stopwords_raw_emp_title', 'count_word_raw_Notes',
       'count_unique_word_raw_Notes', 'count_letters_raw_Notes',
       'count_punctuations_raw_Notes', 'count_words_upper_raw_Notes',
       'count_words_title_raw_Notes', 'count_stopwords_raw_Notes',
       'count_word_raw_purpose', 'count_unique_word_raw_purpose',
       'count_letters_raw_purpose', 'count_punctuations_raw_purpose',
       'count_words_upper_raw_purpose', 'count_words_title_raw_purpose',
       'count_stopwords_raw_purpose', 'noun_countNotes', 'verb_countNotes',
       'adj_countNotes', 'adv_countNotes', 'pron_countNotes']
def prepare_for_logreg_model(train_df, val_df, test_df):
    train_df = transform_categorical_features(train_df)
    val_df = transform_categorical_features(val_df)
    test_df = transform_categorical_features(test_df)
    train_X = train_df.as_matrix()
    val_X = val_df.as_matrix()
    test_X = test_df.as_maxtrix()
    return train_X, val_X, test_X

######################################################################
# Plotting results
# ----------------
#mths_since_last_delinq         4455   mths_since_last_record         6400
# these two numerical variables will be removed as well
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
# I add this to the Data Robot challenge because computational effiency in the time
#  it takes to train these models  will be a part of my decision as to which model
#  I would choose to deploy
#

import time
import math


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))



######################################################################
# Training Process:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Start empty losses array for plotting
#
# creating some important functions



colsinclude =  ['Id', 'is_bad', 'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'pymnt_plan', 'Notes', 'purpose_cat', 'purpose',  'debt_to_income', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths','open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',  'mths_since_last_major_derog', 'policy_code', 'clean_text', 'count_word_raw_emp_title', 'count_unique_word_raw_emp_title', 'count_letters_raw_emp_title', 'count_punctuations_raw_emp_title', 'count_words_upper_raw_emp_title', 'count_words_title_raw_emp_title', 'count_stopwords_raw_emp_title', 'mean_word_len_raw_emp_title', 'word_unique_percent_raw_emp_title', 'punctuation_percent_raw_emp_title', 'lexical_diversity_emp_title', 'count_word_raw_Notes', 'count_unique_word_raw_Notes', 'count_letters_raw_Notes', 'count_punctuations_raw_Notes', 'count_words_upper_raw_Notes', 'count_words_title_raw_Notes', 'count_stopwords_raw_Notes', 'mean_word_len_raw_Notes', 'word_unique_percent_raw_Notes', 'punctuation_percent_raw_Notes', 'lexical_diversity_Notes', 'count_word_raw_purpose', 'count_unique_word_raw_purpose', 'count_letters_raw_purpose', 'count_punctuations_raw_purpose', 'count_words_upper_raw_purpose', 'count_words_title_raw_purpose', 'count_stopwords_raw_purpose', 'mean_word_len_raw_purpose', 'word_unique_percent_raw_purpose', 'punctuation_percent_raw_purpose', 'lexical_diversity_purpose', 'noun_countNotes', 'verb_countNotes', 'adj_countNotes', 'adv_countNotes', 'pron_countNotes']

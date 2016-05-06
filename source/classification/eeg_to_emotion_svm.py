import math
import pandas as pd
import numpy as np
from source.preprocessing import general_preprocess as gp
from source.classification import one_vs_all
from sklearn import preprocessing
from source import config_checker

"""
In this file we do not use valence and arousal predicted from alpha and beta waves to predict emotions. Rather we use
power of alpha and beta waves to predict emotions directly. This alternative approach seems no as good as the one with
valence and arousal
"""

configuration = config_checker.get_configuration()

oversampling = int(configuration.get('oversampling'))
emotion_count = int(configuration.get('emotion_count'))
test_set = float(configuration.get('test_set'))
insight = int(configuration.get('insight'))
one_vs_all = int(configuration.get('one_vs_all'))


def split_data():
    """
    SPlit the data on training and testing set so the individual classes are distributed evenly. Use oversampling if
    flag is set to 1.
    :return: train_x: features of training set
             train_y: targets of training set
             test_x: features of testing set
             test_y: targets of testing set
    """
    if insight:
        eeg_features_df = pd.read_csv('..\\insight_data\\learning_features\\svm_data_all_electrodes.csv')
        electrode_indexes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Emotion']
    else:
        eeg_features_df = pd.read_csv('..\\learning_features\\svm_data_all_electrodes.csv')
        electrode_indexes = ['0', '1', '2', '3', '14', '15', '16', '17', 'Emotion']


    eeg_features_df = eeg_features_df[electrode_indexes]

    eeg_features_df = eeg_features_df.reindex(np.random.permutation(eeg_features_df.index))
    eeg_features_df = eeg_features_df.reset_index(drop=True)

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for i in range(0,emotion_count):
        emotion_df = eeg_features_df[eeg_features_df['Emotion'] == i]
        test_count = math.ceil(emotion_df.shape[0]*test_set)

        train_df = pd.concat([train_df,emotion_df[test_count:]],axis=0)
        test_df = pd.concat([test_df,emotion_df[:test_count]],axis=0)

    train_df = train_df.reset_index(drop=True)
    if oversampling == 1:
        train_df = gp.over_sampling(train_df, gp.histogram(train_df, emotion_count), emotion_count)

    train_y = train_df['Emotion']
    train_x = train_df.drop('Emotion', 1)

    test_y = test_df['Emotion']
    test_x = test_df.drop('Emotion', 1)

    train_x, test_x = gp.feature_scaling(train_x, test_x)

    test_lent = test_x.shape[0]
    df = pd.DataFrame()
    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)
    df = pd.concat([train_x, test_x], axis=0)

    df = preprocessing.normalize(df)
    train_x = df[:(df.shape[0]-test_lent)]
    test_x = df[(df.shape[0]-test_lent):]

    return train_x,train_y,test_x,test_y


def eeg_to_emotion():
    """
    Use this method to predict emotions directly from power fo alpha and beta waves without using valence and arousal
    values. You can choose between single svm classier or one vs all method by changing one_vs_all value in config.ini
    file.
    :return:
    """
    train_x, train_y, test_x, test_y = split_data()

    if one_vs_all:
        one_vs_all.one_vs_all(train_x, train_y, test_x, test_y)
    else:
        gp.my_svm(train_x, train_y, test_x, test_y)
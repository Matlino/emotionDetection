import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.externals import joblib
from source import config_checker
from sklearn.metrics import mean_squared_error

"""
In this file we use power of alpha and beta waves from eeg signal in DEAP dataset. We use this values to train linear
classifiers. The targets are valence and arousal values which participants answered in questionnares in DEAP experiment.
When classifiers are trained we used them on our EEG data to predict valence and arousal values so we can later used
them to classify emotions. We use DEAP data to train classifiers because we dont have enough data to train it.
"""

configuration = config_checker.get_configuration()

#only  electrodes F3, AF3, F4, AF4
electrode_indexes = ['0', '1', '2', '3', '14', '15', '16', '17']

insight = int(configuration.get('insight'))


def prepare_data():
    """
    Read the features (power of alpha and beta waves) and targets (emotions) from files.
    :return features and targets for classifiers:
    """
    features = pd.read_csv(configuration.get('deap_dwt_path'))

    features = features[electrode_indexes]

    target = pd.read_csv('..\\regression_eeg_to_subj\\y.csv')

    return features, target


def select_important_features(features, target):
    """
    When you use many electrodes you can try to select just few of them with the biggest variance as the features for
    classifiers. We are no using it right as it give as worse performance.
    :param features: features only from electrodes with biggest variance
    :param target:
    :return:
    """
    features += 1
    features = SelectKBest(f_regression).fit_transform(features, target['Valence'])
    features -= 1

    features = pd.DataFrame(features)
    target = pd.DataFrame(target)

    return features, target


def regression_train(features ,target):
    """
    Train linera regression classifiers and save them to pickles to be able to use them in application.
    :param features:
    :param target:
    :return: arousal_classifier, valence_classifier: trained classifiers
    """
    # feature scaling
    std_scale = preprocessing.StandardScaler().fit(features)
    features = std_scale.transform(features)

    features = preprocessing.normalize(features)

    # training
    regr = linear_model.LinearRegression()
    regr.fit(features, target['Arousal'])
    arousal_classifier = regr

    regr = linear_model.LinearRegression()
    regr.fit(features, target['Valence'])
    valence_classifier = regr

    # save clfs
    joblib.dump(arousal_classifier, 'classifiers\\arousal_classifier.pkl')
    joblib.dump(valence_classifier, 'classifiers\\valence_classifier.pkl')

    return arousal_classifier, valence_classifier


def regression_predict(arousal_classifier, valence_classifier):
    """
    In this method we use trained classifiers to predict valence and arousal from our EEG data. We then save them to csv
    file for later use.
    :param arousal_classifier:
    :param valence_classifier:
    :return:
    """
    if insight:
        target_path = '..\\insight_data\\extracted_data\\answers\\answers.csv'
        target_df = pd.read_csv(target_path)
        emotiv_data_df = pd.read_csv('..\\insight_data\\learning_features\\svm_data_all_electrodes.csv')
        electrode_indexes = ['0', '1', '2', '3', '4', '5', '6', '7']
        emotiv_data_df = emotiv_data_df[electrode_indexes]
    else:
        target_path = '..\\extracted_data\\answers\\answers.csv'
        target_df = pd.read_csv(target_path)
        emotiv_data_df = pd.read_csv('..\\learning_features\\svm_data_all_electrodes.csv')
        #only  F3, AF3, F4, AF4 electrodes
        electrode_indexes = ['0', '1', '2', '3', '14', '15', '16', '17']
        emotiv_data_df = emotiv_data_df[electrode_indexes]

    # CAR
    for row in range(0, emotiv_data_df.shape[0]):
        emotiv_data_df.iloc[row] = emotiv_data_df.iloc[row] - emotiv_data_df.iloc[row].mean()

    # normalization
    emotiv_data_df = preprocessing.normalize(emotiv_data_df)

    valence_df = pd.DataFrame(valence_classifier.predict(emotiv_data_df), columns=['Valence'])
    arousal_df = pd.DataFrame(arousal_classifier.predict(emotiv_data_df), columns=['Arousal'])

    print("Acrousal score: ", arousal_classifier.score(emotiv_data_df, target_df['Arousal']),
          "mean sq error: ", mean_squared_error(target_df['Arousal'], arousal_classifier.predict(emotiv_data_df)))
    print("Valence score: ", valence_classifier.score(emotiv_data_df , target_df['Valence']),
          "mean sq error: ", mean_squared_error(target_df['Valence'], valence_classifier.predict(emotiv_data_df)))

    df = pd.concat([valence_df, arousal_df, target_df[['Emotion', 'Video']]], axis=1)

    if insight:
        df.to_csv("..\\insight_data\\learning_features\\predicted_subjective.csv", index=False)
    else:
        df.to_csv("..\\regression_eeg_to_subj\\predicted_subjective.csv", index=False)


def eeg_to_subjective():
    """
    You can use this method to run all above at once
    :return:
    """
    features, target = prepare_data()

    # select_important_features()

    arousal_classifier, valence_classifier = regression_train(features, target)

    regression_predict(arousal_classifier, valence_classifier)

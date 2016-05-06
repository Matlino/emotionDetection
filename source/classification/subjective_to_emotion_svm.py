import math
import pandas as pd
import numpy as np
from source.classification import one_vs_all
from source.preprocessing import general_preprocess as gp
from source.plots import confusion_matrix
from sklearn import preprocessing
from sklearn.externals import joblib
from source import config_checker
from sklearn import cross_validation
from sklearn import svm

"""
In this file we are trying to classify emotions by using arousal and valence values which we predicted from eeg signal
using linear regression. We use SVM algorithm to classify individual emotions. We end up using crossvalidation as
a metric because of the lack of data.
"""


# valence aorusal form london eeg eeg
epoc_subjective_path = '..\\regression_eeg_to_subj\\predicted_subjective.csv' #not all all electrodes

insight_subjective_path = '..\\insight_data\\learning_features\\predicted_subjective.csv'

configuration = config_checker.get_configuration()

oversampling = int(configuration.get('oversampling'))
insight = int(configuration.get('insight'))
one_vs_all = int(configuration.get('one_vs_all'))
test_set = float(configuration.get('test_set'))
emotion_count = int(configuration.get('emotion_count'))
is_crossvalidation = int(configuration.get('crossvalidation'))
feature_scaling = int(configuration.get('feature_scaling'))
plot_confusion_matrix = int(configuration.get('plot_confusion_matrix'))
save_clf = int(configuration.get('save_clf'))
video_count = int(configuration.get('video_count'))


def create_features():
    """
    From valence and arousal we create another features as we want to increase options of SVM to differentiate
    individual emotions. Created features are - minimum and maximum arousal and valence per video, mean arousal and
    valence per video, squared arousal and valence.
    :return:
    """
    if insight:
        data = pd.read_csv(insight_subjective_path)
    else:
        data = pd.read_csv(epoc_subjective_path)

    for video_index in range(1, video_count+1):
            video_df = data[data['Video'] == video_index]

            count = video_df.__len__()
            arousal_sum = sum(video_df['Arousal'])
            valence_sum = sum(video_df['Valence'])

            data.loc[data['Video'] == video_index, 'Mean_video_arousal'] = arousal_sum / count
            data.loc[data['Video'] == video_index, 'Mean_video_valence'] = valence_sum / count

            arousal_min = min(video_df['Arousal'])
            valence_min = min(video_df['Valence'])
            arousal_max = max(video_df['Arousal'])
            valence_max = max(video_df['Valence'])

            data.loc[data['Video'] == video_index, 'Min_video_arousal'] = arousal_min
            data.loc[data['Video'] == video_index, 'Min_video_valence'] = valence_min
            data.loc[data['Video'] == video_index, 'Max_video_arousal'] = arousal_max
            data.loc[data['Video'] == video_index, 'Max_video_valence'] = valence_max

    data['Arousal^2'] = data['Arousal']**2
    data['Valence^2'] = data['Valence']**2

    return data


def prepare_data(eeg_features_df):
    """
    Split data to training set and testing set and use oversampling if flag is set to 1.
    :param eeg_features_df:
    :return:
    """
    eeg_features_df = eeg_features_df.reindex(np.random.permutation(eeg_features_df.index))

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for i in range(0, emotion_count):
        emotion_df = eeg_features_df[eeg_features_df['Emotion'] == i]
        test_count = math.ceil(emotion_df.shape[0]*test_set)

        train_df = pd.concat([train_df, emotion_df[test_count:]], axis=0)
        test_df = pd.concat([test_df, emotion_df[:test_count]], axis=0)

    train_df = train_df.reset_index(drop=True)
    if oversampling == 1:
        train_df = gp.over_sampling(train_df, gp.histogram(train_df, emotion_count), emotion_count)

    train_y = train_df['Emotion']
    train_x = train_df.drop(['Emotion', 'Video'], 1)

    test_y = test_df['Emotion']
    test_x = test_df.drop(['Emotion', 'Video'], 1)

    train_x, test_x = gp.feature_scaling(train_x, test_x)

    return train_x, train_y, test_x, test_y


def classes_accuracy(true_values, predictions):
    """
    Calculate accuracy of individual classes
    :param true_values:
    :param predictions: values predicted by SVM
    :return:
    """
    sample_count = np.zeros(emotion_count)
    right_pred_count = np.zeros(emotion_count)
    for true, pred in zip(true_values, predictions):
        if true == pred:
            right_pred_count[true] += 1
        sample_count[true] += 1

    return right_pred_count / sample_count


def svm_classification(eeg_features_df):
    """
    Training and testing SVM classifier. Plot Confusion matrix from best classifier if flag is set to 1.
    :param eeg_features_df:
    :return:
    """
    total_acc = 0
    cycles = 1
    all_best_acc = 0
    for x in range(0, cycles):
        train_x, train_y, test_x, test_y = prepare_data(eeg_features_df)

        # classification
        plot_confusion_matrix = 0
        best_acc, prediction = gp.my_svm(train_x, train_y, test_x, test_y, plot_confusion_matrix)
        total_acc += best_acc
        if best_acc > all_best_acc:
            all_best_acc = best_acc
            best_prediction = prediction

    test_y = np.array(test_y)
    print("MEAN Accuracy: ", total_acc/cycles, " %")
    print("BEST Accuracy: ", all_best_acc, " %")

    classes_accuracy(test_y, best_prediction)

    if confusion_matrix:
        confusion_matrix.prepare_plot(test_y, best_prediction)


def crossvalidation(x, y):
    """
    Cross validation metric. Also plot confusion matrix and save cls if flags are set to 1.
    :param x: features (valence, arousal)
    :param y: target (emotion)
    :return:
    """
    c_array = np.logspace(0, 3, 4)
    gamma_array = np.logspace(-3, 3, 7)

    # feature scaling
    if feature_scaling:
        std_scale = preprocessing.StandardScaler().fit(x)
        x = std_scale.transform(x)

    for c in c_array:
        for gamma in gamma_array:
            clf = svm.SVC(kernel='linear', C=c, gamma=gamma) #kernel= rbf #kernel= poly #kernel= linear
            scores = cross_validation.cross_val_score(clf, x, y, cv=3)
            print("Accuracy: %0.2f (+/- %0.2f) %f %f" % (scores.mean(), scores.std() * 2, c, gamma))
            pred = cross_validation.cross_val_predict(clf, x, y, cv=3)
            print("Classes accuracy: ", classes_accuracy(y, pred))

    print(np.array(y))
    print(pred)

    #plot last one, not best, CARE!!!
    if plot_confusion_matrix:
        confusion_matrix.prepare_plot(y, pred)

    if save_clf:
        clf.fit(x, y)
        joblib.dump(clf, 'classifiers\\'+configuration.get('clf_name')+'.pkl')


def cross_oversampling(eeg_features_df):
    if oversampling == 1:
        eeg_features_df = gp.over_sampling(eeg_features_df, gp.histogram(eeg_features_df, emotion_count), emotion_count)
    return eeg_features_df


def subjective_to_emotion_svm():
    """
    Use this method to run testing of SVM classifiers.
    :return:
    """
    eeg_features_df = create_features()

    if one_vs_all:
        train_x, train_y, test_x, test_y = prepare_data(eeg_features_df)
        one_vs_all.one_vs_all(train_x, train_y, test_x, test_y)
    elif is_crossvalidation:

        # optional - remove some classes in order to increase acc. of classifier
        # eeg_features_df = eeg_features_df[eeg_features_df['Emotion'] != 3]
        # eeg_features_df = eeg_features_df[eeg_features_df['Emotion'] != 4]
        # eeg_features_df = eeg_features_df[eeg_features_df['Emotion'] != 5]
        # eeg_features_df = eeg_features_df[eeg_features_df['Emotion'] != 6]
        if oversampling:
            eeg_features_df = cross_oversampling(eeg_features_df)
        crossvalidation(eeg_features_df.drop(['Emotion', 'Video'], 1), eeg_features_df['Emotion'])
    else:
        svm_classification(eeg_features_df)








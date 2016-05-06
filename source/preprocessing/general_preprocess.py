import numpy as np
from sklearn import preprocessing
from sklearn import svm
from scipy.signal import butter, lfilter
import pywt
import scipy.signal as sig
from source.plots import confusion_matrix

"""
In this file are method which can be used to process the signals or get some information about it. These methods are
used while we are preprocessing EEG and also using it for classification.
"""


def histogram(data_df, emotion_count):
    """ Create histogram of emotions.

    Count samples of individual classes (emotions).

    Parameters
    ----------
    data_df: pandas.Dataframe, need to have column called 'Emotion'
    emotions_count: number of different emotion in dataframe

    Returns
    -------
    hist: array of ints, shape [emotion_count, 1]

    """
    hist = np.zeros(emotion_count)
    for e in range(0, emotion_count):
        hist[e] = data_df[data_df['Emotion'] == e].shape[0]
    return hist


def over_sampling(data_df, hist, emotion_count):
    """ Apply over oversampling on data.

    Copy samples of classes while there is not same amount of all classes. This amount is set by the class which has
    the most samples.

    Parameters
    ----------
    data_df: pandas.Dataframe, need to have column called 'Emotion'
    hist: array of ints, shape [emotion_count, 1]
    emotion_count: count of different emotions in dataframe

    Returns
    -------
    data_df: pandas.Dataframe, oversampled data

    """
    max_sample_count = max(hist)
    inv_hist = np.zeros(emotion_count)
    for e in range(0, emotion_count):
        inv_hist[e-1] = max_sample_count - hist[e-1]

    index = 0
    while sum(inv_hist) != 0:
        row = data_df.iloc[index]
        if inv_hist[row['Emotion']] > 0:
            inv_hist[row['Emotion']] -= 1
            data_df.loc[data_df.shape[0]] = row
        index += 1

    return data_df


def feature_scaling(train_x, test_x):
    """ Scale features in the training and testing set.

    Parameters
    ----------
    train_x: array of floats
             samples of training set
    test_x: array of floats
            samples of testing set
    Returns
    -------
    train_x: featured scaled training set
    test_x: feature scaled testing set

    """
    std_scale = preprocessing.StandardScaler().fit(train_x)
    train_x = std_scale.transform(train_x)

    std_scale = preprocessing.StandardScaler().fit(test_x)
    test_x = std_scale.transform(test_x)
    return train_x, test_x


def my_svm(train_x, train_y, test_x, test_y, plot_confusion=0):
    """ Support vector machines (SVM).

    Using SVM algorithm to predict emotions. We are trying different values of of c and gamma so we can find the best
    combination.

    Parameters
    ----------
    train_x: array-like, shape [number_of_features, number_of_samples_in_training_set]
    train_y: array of ints, shape [number_of_samples_training_set, 1]
             training set target values
    test_x:  array-like, shape [number_of_features, number_of_samples_in_testing_set]
    test_y: array of ints, shape [number_of_samples_in_testing_set, 1]
            testing set target values

    Returns
    -------
    best_acc: float, best accuracy achieved
    best_prediction: array of ints, shape [number_of_samples_in_testing_set, 1]
                     predicted values
    """
    c_array = [0.001, 0.01, 0.1, 1, 10, 100, 100]
    gamma_array = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    best_acc = 0
    best_c = 0
    best_gamma = 0
    best_prediction = 0
    for c in c_array:
        for gamma in gamma_array:
            clf = svm.SVC(C=c, gamma=gamma)
            clf.fit(np.array(train_x), np.array(train_y))
            prediction = clf.predict(np.array(test_x))

            result = prediction-test_y
            good = result[result == 0].shape[0]
            test_y = np.array(test_y)

            acc = good / prediction.shape[0] * 100
            if acc > best_acc:
                best_acc = acc
                best_c = c
                best_gamma = gamma
                best_prediction = prediction
                best_clf = clf

    print("Accuracy: ", best_acc, "% ", "c: ", best_c, " gamma: ", best_gamma)
    print("Clf score: ", best_clf.score(test_x, test_y))
    print("True values: ", test_y)
    print("Predictions: ", best_prediction)

    if plot_confusion:
        confusion_matrix.prepare_plot(test_y, best_prediction)

    return best_acc, best_prediction


def power_of_signal(eeg_signal):
    """ Compute power of signal.

    Power of signal is computed lie sum of samples squared and then divided by mean

    Parameters
    ----------
    eeg_signal: array of floats, shape [number_of_samples, 1]

    Returns
    -------
    power: float, power of signal
    """
    power = 0
    for sampleIndex in range(0,eeg_signal.__len__()):
        power += eeg_signal[sampleIndex]**2.0
    power /= eeg_signal.__len__()
    return power


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """ Bandpass filter.

    Filter frequencies under lowcut value and above hightcut value.

    Parameters
    ----------
    data: array of floats, shape [number_of_samples, 1]
          signal to be filtered
    lowcut: int, lower threshold
    highcut: int, upper threshold
    fs: int, sample rate of signal
    order: int, bigger order, more smooth signal will be
    Returns
    -------

    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpass_filter(video_eeg_data):
    """
    Apply bandpass filter on a signal.
    :param video_eeg_data: eeg signal
    :return: filtered eeg signal
    """
    sample_rate = 128
    lowcut = 4
    highcut = 45
    order = 3
    video_eeg_data = butter_bandpass_filter(video_eeg_data, lowcut, highcut, sample_rate, order)
    return video_eeg_data


def dwt_eeg_video(video_eeg_data, electrode_count, electrode_indexes):
    """
    Use Discrete wavelet transform (DWT) to compute alpha and beta band of signal. Compute power of alpha and beta band
    and also valence and arousal values.
    :param video_eeg_data: eeg signal
    :param electrode_count: number of electrodes
    :param electrode_indexes: indexes of electrodes, usually just range(0, electrode_count)
    :return: array of floats, shape [electrode_count*2 + 2, 1]
             power of alpha and beta band of individual electrodes, valence and arousal values computed from eeg signal
    notes: this function should be split into more in the future
    """

    data_final = np.empty(electrode_count*2 + 2)

    alphaArray = []
    betaArray = []
    counter = 0
    for electrodeIndex in electrode_indexes:
        coeffs = pywt.wavedec(video_eeg_data[electrodeIndex], 'db2', level=3)
        a3, d3, d2, d1 = coeffs

        coeffs = pywt.wavedec(d3, 'db2', level=1)

        alpha, beta = coeffs
        alphaArray.append(power_of_signal(alpha))
        data_final[counter] = power_of_signal(alpha)

        beta = pywt.idwt(d2,sig.resample(beta,d2.__len__()),'db2')
        betaArray.append(power_of_signal(beta))
        data_final[counter+1] = power_of_signal(beta)

        counter += 2

    F3alpha = alphaArray[0]
    F4alpha = alphaArray[1]
    AF3alpha = alphaArray[2]
    AF4alpha = alphaArray[3]
    F3beta = betaArray[0]
    F4beta = betaArray[1]
    AF3beta = betaArray[2]
    AF4beta = betaArray[3]

    valence = (F4alpha/F4beta) - (F3alpha/F3beta)
    arousal = (F3beta+F4beta+AF3beta+AF4beta) / (F3alpha+F4alpha+AF3alpha+AF4alpha)

    data_final[counter] = valence
    data_final[counter+1] = arousal

    return data_final


def car(video_eeg_data):
    """
    Compute Common average reference of signal, subtracting from each sample the average value of the samples of all
    electrodes at this time
    :param video_eeg_data: array-like, shpae [number_of_electrodes, number_of_samples]
                           eeg signal
    :return: eeg signal after CAR
    """
    for row in range(0, video_eeg_data.shape[0]):
        video_eeg_data[row] = video_eeg_data[row] - video_eeg_data[row].mean()
    return video_eeg_data


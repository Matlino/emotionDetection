import pandas as pd
import numpy as np
import os
from source.preprocessing import general_preprocess
from sklearn.externals import joblib
from sklearn import preprocessing
from source import config_checker

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
configuration = config_checker.get_configuration()


def prepare_data(eeg_file_path):
    """ Prepare EEG data to use them in classifiers.

    This method read eeg file, extract signal of electrodes F3, AF3, F4, AF4. Signal from these electrodes is averaged
    by common average reference. Discrete wavelet transform is then use to compute power of alpha and beta waves in
    singal. This values are then normalized return.

    Parameters
    ----------
    eeg_file_path: string, path to eeg file

    Returns
    -------
    eeg_dwt_df: array of floats, shape [8,1]
                power of alpha and beta waves from electrodes F3, F4, AF3, AF4

    """

    eeg_df = pd.DataFrame(pd.read_csv(eeg_file_path, index_col=False))
    electrode_count = 4
    electrode_indexes = [" 'F3'", " 'AF3'", " 'F4'", " 'AF4'"]
    eeg_df = eeg_df[electrode_indexes]

    eeg_df = np.array(eeg_df)
    eeg_df = general_preprocess.car(eeg_df)

    eeg_dwt_df = general_preprocess.dwt_eeg_video(eeg_df, electrode_count, range(0, 4))

    eeg_dwt_df = preprocessing.normalize(eeg_dwt_df)

    return eeg_dwt_df[0]


def predict_valence_arousal(eeg_dwt_df):
    """ Predict valence and arousal values.

    Method use valence and arousal classifiers stored in classifiers directory and precit valence and arousal values
    from power of alpha and beta waves.

    Parameters
    ----------
    eeg_dwt_df: Power of alpha and beta waves.

    Returns
    -------
    valence: float, predicted valence
    arousal: float, predicted
    eeg_dwt_df: array of floats, shape [8,1]
                power of alpha and beta waves from electrodes F3, F4, AF3, AF4
    """

    arousal_clf = joblib.load(os.path.join(BASE_DIR, 'classifiers', configuration.get('arousal_classifier')))
    valence_clf = joblib.load(os.path.join(BASE_DIR, 'classifiers', configuration.get('valence_classifier')))

    arousal = arousal_clf.predict(eeg_dwt_df[:-2])
    valence = valence_clf.predict(eeg_dwt_df[:-2])

    eeg_dwt_df[-2] = valence
    eeg_dwt_df[-1] = arousal
    eeg_dwt_df = np.append(eeg_dwt_df, arousal**2)
    eeg_dwt_df = np.append(eeg_dwt_df, valence**2)

    return arousal, valence, eeg_dwt_df


def predict_emotion(eeg_file_path):
    """ Prepare data and detect emotion .

    This method use our classifiers to predict valence, arousal and emotion.

    Parameters
    ----------
    eeg_file_path : string, path to the eeg file.

    Returns
    -------
    arousal float
    valence float
    emotion int, in the range 0 - 6
        0: happy
        1: sad
        2: disgusted
        3: angry
        4: scared
        5: surprised
        6: neutral
    """

    eeg_dwt_df = prepare_data(eeg_file_path)

    arousal, valence, eeg_dwt_df = predict_valence_arousal(eeg_dwt_df)

    emotion_clf = joblib.load(os.path.join(BASE_DIR, 'classifiers', configuration.get('emotion_classifier')))

    emotion = emotion_clf.predict(eeg_dwt_df)

    return arousal, valence, emotion




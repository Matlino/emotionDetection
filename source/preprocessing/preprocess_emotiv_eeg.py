import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from source.preprocessing import general_preprocess as gp
from source import config_checker
import os

"""
In this file we preprocess EEG data acquired in our experiment using Emotiv EPOC device. Participants watches 20 music
videos but we want eeg date for every video individually so we need to split them. For this task we will also use
timestamps from Tobii studio file where we see in what time which video for presented to the particiapants. In the emotiv
EEG file we see time when recording started and we know that sampling rate of Emotiv EPOC data os 128hz. With this
information we can split the date to the individual videos adn then apply discrete wavelet transform on them to extract
alpha and beta band.
"""

configuration = config_checker.get_configuration()

emotiv_eeg_freq = int(configuration.get('emotiv_eeg_freq'))
video_count = int(configuration.get('video_count'))
insight = int(configuration.get('insight'))


def load_raw_eeg(participants_index):
    """
    Load whole EEG data file and extract important electrodes data
    :param participants_index:
    :return: electrodes eeg data
    """
    if insight:
        eeg_path = os.path.join(configuration.get('insight_raw_data_path'), str(participants_index)+"\\eeg.csv")

        eeg_df = pd.read_csv(eeg_path)
        print("Shape of eeg file: ", eeg_df.shape)

        eeg_df = pd.DataFrame(eeg_df[['COUNTER', 'AF3', 'AF4', 'T7', 'T8']])
    else:
        eeg_path = os.path.join(configuration.get('epoc_raw_data_path'), str(participants_index)+"\\eeg.csv")

        eeg_df = pd.read_csv(eeg_path)
        print("Shape of eeg file: ", eeg_df.shape)

        eeg_df = pd.DataFrame(eeg_df[['COUNTER', 'F3', 'F4', 'AF3', 'AF4', 'F7', 'F8',
                              'FC5', 'FC6', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']])

    return eeg_df


def substract_times(t1,t2):
    """
    Subtract two times in string, solution returned in seconds
    :param t1:
    :param t2:
    :return:
    """
    t1 = time.strptime(t1, "%H:%M:%S")
    t2 = time.strptime(t2, "%H:%M:%S")

    t1 = datetime(*t1[:6])
    t2 = datetime(*t2[:6])

    t3 = t2 - t1
    return t3.seconds


def split_data(extracted_eeg, tobii_time_stamps, participants_index):
    """
    split eeg data into parts, where every part corresponds with one music video participant watched
    :param extracted_eeg:
    :param tobii_time_stamps: timestamps from Tobii Studio
    :param participants_index:
    :return:
    """
    if insight:
        eeg_start_arr = configuration.get('insight_recording_starts').split()
    else:
        eeg_start_arr = configuration.get('epoc_recording_starts').split()

    video_num = 0
    for i in range(0, video_count):
        video_num += 1
        print("Video number: ", video_num)

        if video_num < 10:
            zero = '0'
        else:
            zero = ''

        # extract start of video from tibii data
        temp_df = tobii_time_stamps[tobii_time_stamps['MediaName'] == zero+str(video_num)+".wmv"]
        video_start = temp_df['LocalTimeStamp'].iloc[0]
        video_start = video_start[:8]

        # subtract video start time and egg recording start time
        before_start = substract_times(eeg_start_arr[participants_index-1], video_start)

        # extract eeg data when video was projected
        eeg_blank = before_start * emotiv_eeg_freq

        # eeg_video = extracted_eeg.iloc[eeg_blank-1+(emotiv_eeg_freq*5):eeg_blank-1+(emotiv_eeg_freq*65)]

        # with the 5 second white cross before video
        white_cross = emotiv_eeg_freq*5
        # without
        white_cross = 0
        eeg_video = extracted_eeg.iloc[eeg_blank-1+white_cross:eeg_blank-1+(emotiv_eeg_freq*65)]


        if insight:
            df = pd.DataFrame(eeg_video[['AF3', 'AF4', 'T7', 'T8']])
            # eeg_video_path = "..\\insight_data\\extracted_data\\eeg_videos\\participant"+str(participants_index) + \
            #                  "\\video"+str(video_num)+".csv"
            eeg_video_path = "..\\dvd_data\\eeg\\insight\\participant"+str(participants_index) + \
                             "\\video"+str(video_num)+".csv"

        else:
            df = pd.DataFrame(eeg_video[['F3', 'F4', 'AF3', 'AF4', 'F7', 'F8', 'FC5', 'FC6', 'T7', 'T8', 'P7',
                                         'P8', 'O1', 'O2']])
            eeg_video_path = "..\\extracted_data\\eeg_videos\\participant"+str(participants_index) + \
                             "\\video"+str(video_num)+".csv"
            # eeg_video_path = "..\\dvd_data\\eeg\\epoc\\participant"+str(participants_index) + \
            #                  "\\video"+str(video_num)+".csv"

        df.to_csv(eeg_video_path, index=False)


def load_timestamps(participants_index):
    """
    Load timestamps from Tobii studio
    :param participants_index:
    :return:
    """
    if insight:
        tobii_df = pd.read_csv("..\\insight_data\\extracted_data\\timestamps\\"+str(participants_index)+".csv", sep=';')
    else:
        tobii_df = pd.read_csv("..\\extracted_data\\timestamps\\"+str(participants_index)+".csv", sep=';')

    return tobii_df


def dwt(participants_count, bandpass_filter):
    """
    Apply discrete wavelet transform on data
    :param participants_count:
    :param bandpass_filter:
    :return:
    """
    if insight:
        electrode_indexes = np.arange(0, 4, 1)
    else:
        electrode_indexes = np.arange(0, 14, 1)

    electrode_count = electrode_indexes.__len__()

    data_final = pd.DataFrame()
    for participants_index in range(1, participants_count+1):
        print("Start participant: ", participants_index)

        participants_dwt_data = np.empty([video_count, electrode_count*2 + 2])
        for video_index in range(1, video_count+1):
            if insight:
                video_path = "..\\insight_data\\extracted_data\\eeg_videos\\" \
                         "participant"+str(participants_index)+"\\video"+str(video_index)+".csv"
            else:
                video_path = "..\\extracted_data\\eeg_videos\\" \
                         "participant"+str(participants_index)+"\\video"+str(video_index)+".csv"
            video_eeg_data = pd.read_csv(video_path)

            if bandpass_filter == 1:
                video_eeg_data = gp.bandpass_filter(video_eeg_data)

            video_eeg_data = np.array(video_eeg_data)
            video_eeg_data = gp.car(video_eeg_data)

            video_eeg_data = gp.dwt_eeg_video(video_eeg_data, electrode_count, electrode_indexes)

            participants_dwt_data[video_index-1] = video_eeg_data

        data_final = pd.concat([data_final, pd.DataFrame(participants_dwt_data)], axis=0)

    data_final = data_final.reset_index(drop=True)

    if insight:
        data_final.to_csv('..\\insight_data\\learning_features\\filtered_all_signal_features.csv', index=False)
    else:
        data_final.to_csv('..\\learning_features\\filtered_all_signal_features.csv', index=False)


def prepare_data(participants_count):
    """
    Run method above for every participant
    :param participants_count:
    :return:
    """
    for participants_index in range(1, participants_count+1):
        print("Start participant: ", participants_index)
        tobii_time_stamps = load_timestamps(participants_index)

        eeg_df = load_raw_eeg(participants_index)

        split_data(eeg_df, tobii_time_stamps, participants_index)


# create svm data file for suport vectore machine
def select_best_features(squared_features, select_kbest):
    if insight:
        answers_path = "..\\insight_data\\extracted_data\\answers\\answers.csv"
        eeg_features_path = "..\\insight_data\\learning_features\\filtered_all_signal_features.csv"
    else:
        answers_path = "..\\extracted_data\\answers\\answers.csv"
        eeg_features_path = "..\\learning_features\\filtered_all_signal_features.csv"
    answers_df = pd.read_csv(answers_path)
    eeg_features_df = pd.read_csv(eeg_features_path)

    if squared_features:
        eeg_features_df = pd.concat([eeg_features_df, eeg_features_df*eeg_features_df], axis=1)

    if select_kbest:
        eeg_features_df = SelectKBest(f_regression).fit_transform(eeg_features_df, answers_df['Emotion'])
        eeg_features_df = pd.DataFrame(eeg_features_df)

    eeg_features_df = pd.concat([eeg_features_df, answers_df['Emotion']], axis=1)

    if insight:
        eeg_features_df.to_csv('..\\insight_data\\learning_features\\svm_data_all_electrodes.csv', index=False)
    else:
        eeg_features_df.to_csv('..\\learning_features\\svm_data_all_electrodes.csv', index=False)


def preprocess__emotiv_eeg():
    """
    Use method above by using this method
    :return:
    """
    if insight:
        participants_count = int(configuration.get('insight_participants_count'))
    else:
        participants_count = int(configuration.get('epoc_participants_count'))

    prepare_data(participants_count)

    # dwt(participants_count, 0)
    #
    # select_kbest = 0
    # squared_features = 0
    # select_best_features(squared_features, select_kbest)

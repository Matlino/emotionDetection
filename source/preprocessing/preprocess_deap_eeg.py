import os
import pandas as pd
import numpy as np
from source.preprocessing import general_preprocess as gp
from source import config_checker

"""
In this file are method use to preprocess EEG singal obtained with EEG device which has 32 electrodes. Dataset with this
EEG data was acquired from this website: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/. If you want to use thid methods,
you need to specify path to those data file in config.ini file.
"""

configuration = config_checker.get_configuration()
deap_file_names = os.listdir(configuration.get('deap_data_path'))


def dwt(video_count, electrode_count):
    """
    Compute discrete wavelet transform of data from DEAP dataset. Method compute dwt of signal from all participants and
    use just those electrodes which are located also on Emotiv EPOC device. Path for import and export the data need to
    be set in config.ini file.
    :param video_count: number of videos each participants watched
    :param electrode_count: number of electrodes
    """
    # those are the electrodes which are also used on Emotiv EPOC device
    electrodeIndexes = [2, 19, 1, 17, 3, 20, 4, 21, 7, 25, 11, 29, 13, 31]

    data_final = pd.DataFrame()
    for file_index in range(0, len(deap_file_names)):
        print("Start participant: ", file_index+1)

        pkl = pd.read_pickle(os.path.join(configuration.get('deap_data_path'), deap_file_names[file_index]))

        participant_eeg = pkl['data']

        participants_dwt_data = np.empty([video_count, electrode_count*2 + 2])
        for video_index in range(0, video_count):
            video_eeg_data = participant_eeg[video_index]

            participants_dwt_data[video_index-1] = gp.dwt_eeg_video(video_eeg_data, electrode_count, electrodeIndexes)

        data_final = pd.concat([data_final, pd.DataFrame(participants_dwt_data)], axis=0)

    data_final = data_final.reset_index(drop=True)

    data_final.to_csv(configuration.get('deap_dwt_path'), index=False)


def preprocess_london_eeg():
    video_count = 40
    electrode_count = 14
    dwt(video_count, electrode_count)
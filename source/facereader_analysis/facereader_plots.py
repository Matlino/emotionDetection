import pandas as pd
import numpy as np
import matplotlib
import source.preprocessing.preprocess_emotiv_eeg as prep
import matplotlib.pyplot as plt
import configparser
import matplotlib.patches as mpatches
from source import config_checker
from source import config_checker

"""
In this file are method for creating plots using data from Noldus Facereader.
"""

texts = config_checker.get_texts()
matplotlib.rc('font', family='Arial')

configuration = config_checker.get_configuration()

epoc_participants_count = int(configuration.get('epoc_participants_count'))
emotion_count = int(configuration.get('emotion_count'))


def plot_participants():
    """
    Pie chart for every participants with percentual emotion detection.
    :return:
    """
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    subplots_indexes = [251, 252, 253, 254, 255, 256, 257, 258, 259]
    # fig, axes = plt.subplots(2, 5, figsize=(10,5))
    fig.subplots_adjust()

    labels = emotions_names = texts.get('emotion_names').split()
    # labels = [texts.get('neutral_emotion'), texts.get('anger'), texts.get('other')]
    for participants_index in range(1, epoc_participant_count + 1):
        print("Participant: ", participants_index)
        in_path = "..\\extracted_data\\facereader\\"+str(participants_index)+".csv"
        face_df = pd.read_csv(in_path)
        face_sum = face_df.sum()
        print(face_sum)
        ax = fig.add_subplot(subplots_indexes[participants_index-1])
        sizes = face_sum
        colors = ['lightblue', 'lightcoral', 'yellowgreen', 'mediumorchid', 'goldenrod', 'slategrey', 'brown']

        # patches, texts =
        ax.pie(sizes, shadow=True, startangle=90, colors=colors) # , labels=labels autopct='%1.1f%%',
        ax.set_title(texts.get('participant')+" "+str(participants_index), fontsize=16, fontweight='bold')
        # ax.legend(patches, labels, bbox_to_anchor=(1, 1)) # loc="best"

        ax.axis('equal')

    # prerob do cyklu potom
    patches = []
    patches.append(mpatches.Patch(color='lightblue', label=labels[6]))
    patches.append(mpatches.Patch(color='lightcoral', label=labels[0]))
    patches.append(mpatches.Patch(color='yellowgreen', label=labels[1]))
    patches.append(mpatches.Patch(color='mediumorchid', label=labels[3]))
    patches.append(mpatches.Patch(color='goldenrod', label=labels[5]))
    patches.append(mpatches.Patch(color='slategrey', label=labels[4]))
    patches.append(mpatches.Patch(color='brown', label=labels[2]))

    plt.legend(handles=patches, loc=(1.5, 0.5)) # handles=[red_patch, blue_patch]

    plt.show()


def plot_histogram():
    """
    Histogram of emotions detected by Noldus FaceReader
    :return:
    """
    in_path = '..\\extracted_data\\facereader\\video_emotions.csv'
    emotion_df = pd.read_csv(in_path)
    hist = []
    for label in range(0, emotion_count):
        temp_df = emotion_df[emotion_df['Emotion'] == label]
        hist.append(temp_df.shape[0])

    print(hist)
    results = hist

    # hisotgram of eeg samples
    # results = [51,  22,  40,   5,  10,  19,  33]
    
    emotions_names = texts.get('emotion_names').split()

    i = 0
    for count in results:
        emotions_names[i] = emotions_names[i] + " (" + str(count) + ")"
        i += 1

    fig = plt.figure(figsize=(12, 8))

    plt.rc('ytick') # , labelsize=15
    # plt.rc('xtick', labelsize=16)
    plt.axis((0, emotions_names.__len__()+1, 0, max(results)))

    arr = np.arange(1, emotions_names.__len__()+1, 1)

    ax = fig.add_subplot(111)
    ax.set_xlabel(texts.get('emotions'), fontsize=20)
    ax.set_ylabel(texts.get('sample_count'), fontsize=20)


    plt.bar(arr, results, align='center', color='lightblue', width=0.5)
    # tu nastavujem velkost textu - emotion names
    plt.xticks(arr, emotions_names, rotation=45) # , size=15 r

    plt.show()


def analyze():

    plot_histogram()

    plot_participants()

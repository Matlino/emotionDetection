import pandas as pd
import unicodedata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os
from source import config_checker

"""
In this file we preprocess answers from questionannares from Tobii Studio. We need to remove diacritics, split to
individual columns and label the emotions.
"""

epoc_answer_file_names = os.listdir("..\\extracted_data\\answers")
insight_answer_file_names = os.listdir("..\\insight_data\\extracted_data\\answers")

configuration = config_checker.get_configuration()
insight = int(configuration.get('insight'))


# delete strings like "pokojny", "negativne" etc.
def label_extremes(df):
    for i in range(0,len(df)):
        if len(df.iloc[i][0]) > 1:
            df.iloc[i][0] = df.iloc[i][0][0]
        if len(df.iloc[i][1]) > 1:
            df.iloc[i][1] = df.iloc[i][1][0]
    return df


# split data to three columns - arousal, valence, emotion
def split_columns(df):
    arousal = []
    valence = []
    emotion = []
    for i in range(0,len(df)):
        if i % 3 == 0:
            arousal.append(df.iloc[i][0])
        if i % 3 == 1:
            valence.append(df.iloc[i][0])
        if i % 3 == 2:
            emotion.append(df.iloc[i][0])

    df_array = np.array([arousal, valence, emotion])
    df_array = np.transpose(df_array)

    columns = ['Arousal', 'Valence', 'Emotion']
    return pd.DataFrame(df_array, columns=columns)


# remove diacritics from text
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


#remove diacritics from emotions.csv file and save it
def remove_diacritics(path):
    f = open(path, 'r')
    emotions = f.read()
    emotions = strip_accents(emotions)
    f = open(path, 'w')
    f.write(emotions)
    f.close()
    print("Log: Diacritics removed")
    return emotions


# switch statement for emotion labels
def label_emotion(x):
    return {
        'Radost': 0,
        'Smutok': 1,
        'Znechutenie': 2,
        'Hnev': 3,
        'Strach': 4,
        'Prekvapenie': 5,
        'Neutralna emocia': 6,
    }[x]

# label emotions:
# Radost = 0
# Smutok = 1
# Znechutenie = 2
# Hnev = 3
# Strach = 4
# Prekvapenie = 5
# Neutralna emocia = 6
def labelEmotions(df):
    emotions = df['Emotion']
    for i in range(0,emotions.size):
        emotions[i] = label_emotion(emotions[i])

    df['Emotion'] = emotions

    return df


def add_video_index(answers_df):
    arr = np.arange(1, 21, 1)

    answers_df['Video'] = arr

    return answers_df


# plot 2D Valence-Arousal model
def plot_valence_arousal_model(answers_df):
    texts = config_checker.get_texts()
    matplotlib.rc('font', family='Arial')


    colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'magenta']

    classes = ['Joy','Sadness','Disgust','Anger', 'Fear', 'Surprise', 'Neutral']
    class_colours = colors
    recs = []
    for i in range(0,len(class_colours)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))

    line = np.zeros(10/0.1)
    line.fill(5)


    # single graph with legend
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_xlabel(texts.get('valence'))
    ax.set_ylabel(texts.get('arousal'))


    plt.legend(recs,classes,loc=4)
    plt.xlim(0, 13)
    plt.ylim(0, 10)

    plt.scatter(answers_df['Valence'], answers_df['Arousal'],  color=colors, s=70)
    plt.scatter(np.arange(0, 10, 0.1), line,s=3)
    plt.scatter(line,np.arange(0, 10, 0.1),s=3)

    plt.legend(recs,classes,loc=4)
    plt.xlim(0, 13)
    plt.ylim(0, 10)
    plt.scatter(answers_df['Valence'], answers_df['Arousal'],  color=colors, s=70)
    plt.scatter(np.arange(0, 10, 0.1),line,s=3)
    plt.scatter(line,np.arange(0, 10, 0.1),s=3)

    # plt.show()
    # plt.savefig('..\\plots\\1.png')
    plt.clf()

    subplots_indexes = [241, 242, 243, 244, 245, 246, 247]
    emotion_labels = emotions_names = texts.get('emotion_names').split()
    fig.subplots_adjust(hspace=.3, wspace=.4)
    for i in range(0,7):

        ax = fig.add_subplot(subplots_indexes[i])
        #ax.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
        ax.set_title(emotion_labels[i], fontsize=18, fontweight='bold')
        ax.set_xlabel(texts.get('valence'), fontsize=16)
        ax.set_ylabel(texts.get('arousal'), fontsize=16)

        # plt.legend(recs,classes,loc=4)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        answersDF1 = answers_df[answers_df['Emotion'] == i]


        #set size of diffrents points on graph depends on hwo many same values are there
        size = np.zeros(100)
        for j in range(0,answersDF1.__len__()):
                size[(answersDF1['Valence'].iloc[j]-1)*10 + answersDF1['Arousal'].iloc[j]-1] += 1

        #color=colors[i], pridja aby to bol ofarebne
        plt.scatter(answersDF1['Valence'], answersDF1['Arousal'], color='black',
                    s=40*size[(answersDF1['Valence']-1)*10 + answersDF1['Arousal']-1])
        plt.scatter(np.arange(0, 10, 0.1),line,s=3)
        plt.scatter(line,np.arange(0, 10, 0.1),s=3)

        # plt.savefig('..\\plots\\together.png')
        # plt.savefig('..\\plots\\'+str(i+2)+'.png')
        # plt.clf()


    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    plt.show()


# format answers and label them
def format_answers(file_index):
    if insight:
        path = "..\\insight_data\\extracted_data\\answers\\"+insight_answer_file_names[file_index]
    else:
        path = "..\\extracted_data\\answers\\"+epoc_answer_file_names[file_index]

    remove_diacritics(path)

    answers_df = pd.read_csv(path)

    answers_df = split_columns(answers_df)

    answers_df = labelEmotions(answers_df)

    answers_df = label_extremes(answers_df)

    return answers_df


# preprocessed data from questionnaires
def preprocess_answers():
    all_answers_df = pd.read_csv("C:\\Users\\Matlo\\PycharmProjects\\EmotionAnalysis\\extracted_data\\answers\\answers.csv")

    plot_valence_arousal_model(all_answers_df)

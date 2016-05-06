import pandas as pd
import source.preprocessing.preprocess_emotiv_eeg as prep
from source import config_checker

"""
In this file we analyze data from Noldus FaceReader which detect emotions from face expressions. We were recording face
of participants whole time they were watching 20 music videos. So we need to split data similarly as EEG data. First we
need to clean them because they may be parts were FaceREader could not recognize the face of participants. Then, when,
we split them, we found out which emotions noldus detect most for every video. Afterwards we comapre these emotions to
those from questionnares and compute accuracy of Noldus FaceReader.

Run methods in this file by using method analyze()
"""

configuration = config_checker.get_configuration()
insight = int(configuration.get('insight'))
video_count = int(configuration.get('video_count'))
facereader_freq = int(configuration.get('facereader_freq'))

emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared', 'Disgusted']

if insight:
    participants_count = int(configuration.get('insight_participants_count'))
else:
    participants_count = int(configuration.get('epoc_participants_count'))

def clean_data(facereader_df):
    """
    Remove unnecesarry columns and failed samples replac with 0
    :param facereader_df:
    :return:
    """
    timestamps = facereader_df['Video Time']
    facereader_df = facereader_df.drop(['Video Time', 'Stimulus', 'Event Marker'], 1)

    # check if there are fialed samples and replace them with zeroes
    data_test = facereader_df[emotions].dtypes == 'float64'
    if data_test[data_test == False].shape[0] > 0:
        facereader_df[(facereader_df['Neutral'] == 'FIND_FAILED') | (facereader_df['Neutral'] == 'FIT_FAILED')] = 0

    # convert to float
    facereader_df[emotions] = facereader_df[emotions].astype(float)
    facereader_df = pd.concat([timestamps, facereader_df], axis=1)

    return facereader_df


def split_data(facereader_df, participants_index):
    """
    Split data to parts coresponding to videos
    :param facereader_df:
    :param participants_index:
    :return:
    """
    if insight:
        in_path = "..\\insight_data\\extracted_data\\timestamps\\"+str(participants_index)+".csv"
        out_path = "..\\insight_data\\extracted_data\\facereader\\"+str(participants_index)+".csv"
        facereader_start = ['19:42:35', '20:19:41', '20:59:14']
    else:
        in_path = "..\\extracted_data\\timestamps\\"+str(participants_index)+".csv"
        out_path = "..\\extracted_data\\facereader\\"+str(participants_index)+".csv"
        facereader_start = ['15:34:26', '13:21:44', '15:09:03', '14:07:14', '15:26:55', '12:02:59',
                            '13:05:23', '14:41:53', '16:41:35']

    tobii_time_stamps = pd.read_csv(in_path, sep=';')
    facereader_participant_df = pd.DataFrame()


    for video_num in range(1, video_count + 1):
        print("Video number: ", video_num)

        if video_num < 10:
            zero = '0'
        else:
            zero = ''

        # extract start of video from tibii data
        temp_df = tobii_time_stamps[tobii_time_stamps['MediaName'] == zero+str(video_num)+".wmv"]
        video_start = temp_df['LocalTimeStamp'].iloc[0]
        video_start = video_start[:8]

        # subtract video start time and facereader recording start time
        before_start = prep.substract_times(facereader_start[participants_index-1], video_start)

        # extract facereader data when video was projected
        facereader_blank = before_start * facereader_freq



        # with the 5 second white cross before video
        white_cross = facereader_freq*5
        # without
        white_cross = 0
        facereader_video_df = facereader_df.iloc[facereader_blank-1+white_cross: facereader_blank-1+
                                                                                         (facereader_freq*65)]


        face_video_path = "..\\dvd_data\\facereader\\with_insight\\participant"+str(participants_index) + \
                             "\\video"+str(video_num)+".csv"
        facereader_video_df.to_csv(face_video_path, index=False)

        facereader_video_df = pd.DataFrame(facereader_video_df[emotions].sum()).transpose()
        facereader_participant_df = pd.concat([facereader_participant_df, facereader_video_df], axis=0)

    facereader_participant_df.to_csv(out_path, index=False)


def compute_emotions():
    """
    Comprute which emotion was detected most for every video
    :return:
    """
    emotions_names = ['Happy', 'Sad', 'Disgusted', 'Angry', 'Scared', 'Surprised', 'Neutral']
    final_labels_df = pd.DataFrame(dtype=int)

    for participants_index in range(1, participants_count + 1):
        labels_df = pd.DataFrame(index=range(0, 20), columns=['Emotion'], dtype=int)
        print("Participant: ", participants_index)
        in_path = "..\\extracted_data\\facereader\\"+str(participants_index)+".csv"
        face_df = pd.read_csv(in_path)
        maxiumums = face_df.max(axis=1)

        for idx, emotion in enumerate(emotions_names):
            # print(maxiumums == face_df[emotion])
            match_values_df = (maxiumums == face_df[emotion])
            # print(match_values_df[match_values_df == True].index.tolist())
            true_indexes = match_values_df[match_values_df == True].index.tolist()
            labels_df.iloc[true_indexes] = idx

        final_labels_df = pd.concat([final_labels_df, labels_df], axis=0)
        # print(labels_df)

    final_labels_df = final_labels_df.reset_index(drop=True)
    outpath = '..\\extracted_data\\facereader\\video_emotions.csv'
    final_labels_df.to_csv(outpath, index=False)


def measure_accuracy():
    """
    Compare detected emotions with those from questionnares and compute accuracy.
    :return:
    """
    facereader_in_path = '..\\extracted_data\\facereader\\video_emotions.csv'
    questionnares_in_path = '..\\extracted_data\\answers\\answers.csv'
    facereader_df = pd.read_csv(facereader_in_path)
    questionnares_df = pd.read_csv(questionnares_in_path)

    sample_count = facereader_df.shape[0]

    emotion_df = pd.DataFrame(index=range(0,180), columns=['Emotion'])
    emotion_df['Emotion'] = questionnares_df['Emotion']

    compared_df = facereader_df == emotion_df
    good_pred = compared_df[compared_df['Emotion'] == True]
    facereader_accuracy = good_pred.shape[0] / sample_count
    print("Facereader accuracy is: ", facereader_accuracy*100, "%")


def analyze():
    """
    Use this method to use those above
    :return:
    """
    for participants_index in range(1, participants_count+1):
        # participants_index = 4
        print("Starting analyze participant: ", participants_index)
        if insight:
            path = "..\\insight_data\\raw_data\\"+str(participants_index)+"\\facereader.txt"
        else:
            path = "..\\raw_data\\"+str(participants_index)+"\\facereader.txt"

        facereader_df = pd.read_csv(path, sep='\t')

        facereader_df = clean_data(facereader_df)

        split_data(facereader_df, participants_index)

    measure_accuracy()






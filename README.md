# Emotion detection from EEG signal recorded by Emotiv EPOC device

Source code provided here is part of my bachelor thesis in which we propose a method for emotion detection from EEG signal. We used several machine learning techniques and trained classiefrs in application.

To train classifiers we needed data so we conducted experiment. In experiment participants watched music videos while we were recording their EEG signal and face expressions. Participants also answer questions about their emotions so we can label the eeg data. We record face expressions in order to compare our emotion detection method with Noldus FaceReader, which is tool to detect emotions from face expressions. After experiment we [preprocess](source/preprocessing) in order to [classify]((source/classification)) emotions using machine learning techniques.

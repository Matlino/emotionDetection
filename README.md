# Emotion detection from EEG signal recorded by Emotiv EPOC device

Source code provided here is part of my bachelor thesis in which we propose a method for emotion detection from EEG signal. We used several machine learning techniques and trained classiefrs in application. Application can be started by running app.py located [here](source/application).

To train classifiers we needed data so We conducted experiment. In experiment participants watched music videos while we were recording their EEG signal and face expressions. Participants also answers questions about their emotions so we can label the eeg data. We record face expressions in order to compare are method with Noldus FaceReader, which is tool to detect emotions from face expressions. After experiment we [preprocess](source/preprocessing) in order to [classify]((source/classification)) emotions using machine learning techniques.

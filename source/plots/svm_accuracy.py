import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from source import config_checker


def plot_classes_accuracy():
    #if font isnt change matplot dont display some characters corretcly
    matplotlib.rc('font', family='Arial')

    texts = config_checker.get_texts()

    results = [71, 43, 29, 21, 21, 7, 32]
    emotions_names = texts.get('emotion_names').split()
    emotions_names[6] = "Spolu"
    i = 0
    for acc in results:
        emotions_names[i] = emotions_names[i] + " " + str(acc) + "%"
        i += 1

    fig = plt.figure(figsize=(12, 8))

    plt.rc('ytick') # , labelsize=15
    # plt.rc('xtick', labelsize=16)
    plt.axis((0, emotions_names.__len__()+1, 0, 100))


    arr = np.arange(1, emotions_names.__len__()+1, 1)


    ax = fig.add_subplot(111)
    ax.set_xlabel(texts.get('emotions'), fontsize=20)
    ax.set_ylabel(texts.get('accuracy')+" (%)", fontsize=20)


    plt.bar(arr, results, align='center', color='lightblue', width=0.5)
    # tu nastavujem velkost textu - emotion names
    plt.xticks(arr, emotions_names, rotation=45) # , size=15 r

    plt.show()
    pass


plot_classes_accuracy()


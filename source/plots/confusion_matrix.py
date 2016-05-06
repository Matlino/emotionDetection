import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from source import config_checker
from sklearn.metrics import confusion_matrix

texts = config_checker.get_texts()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    matplotlib.rc('font', family='Arial')
    emotions_names = texts.get('emotion_names').split()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
   #  plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(emotions_names))
    plt.xticks(tick_marks, emotions_names, rotation=45)
    plt.yticks(tick_marks, emotions_names)
    plt.tight_layout()
    plt.ylabel(texts.get('true_label'), fontsize=20)
    plt.xlabel(texts.get('predicted_label'), fontsize=20)


def prepare_plot(y_test, y_pred):

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print(texts.get('confusion_matrix_without_normalization'))
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, title=texts.get('confusion_matrix_without_normalization'))

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(texts.get('normalized_confusion_matrix'))
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title=texts.get('normalized_confusion_matrix'))

    plt.show()

from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support

"""
In this files is our one vs all method as we are using it. We used as a alternative method in order to achieve better
performance but it did not work.
"""


def one_vs_all(train_x, train_y, test_x, test_y):
    """
    Apply one vs all strategy in order to predict emotions.
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return:
    """
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    classifier.fit(train_x, train_y)
    pred = classifier.predict(test_x)

    # print out precission and recall metrics
    print(precision_recall_fscore_support(test_y, pred, average='macro'))
    print(precision_recall_fscore_support(test_y, pred, average='micro'))
    print(precision_recall_fscore_support(test_y, pred, average='weighted'))
    print(precision_recall_fscore_support(test_y, pred, average=None))



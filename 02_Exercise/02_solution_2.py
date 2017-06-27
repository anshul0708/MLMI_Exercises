from __future__ import division

import os
import numpy
import pandas
import cv2
import random
import matplotlib.pyplot as plt
import warnings

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Ignore some warnings
warnings.filterwarnings("ignore")

# Path to dataset
DIGITS_PATH = os.getcwd() + '/03-digits-dataset/'

""" Task A """

# Read train data into numpy array
DIGITS_DATASET = pandas.read_csv(DIGITS_PATH + 'train.csv')

X = DIGITS_DATASET.ix[:,1:].values.astype(numpy.uint8)
y = DIGITS_DATASET['label'].values.astype(numpy.uint8)

def taska():
    # Test a few images
    for i in range(0,4):
        index = random.randint(0,1000)
        cv2.imshow("image", X[index].reshape(28,28)), cv2.waitKey(0)
        print y[index]

#taska()

""" Task B """

pca = PCA(n_components=0.8)
X_transformed = pca.fit_transform(X)

#print X_transformed.shape
#print pca.explained_variance_ratio_
#print pca.n_components_

""" Task C """

def label_list(y, number):
    """ Create new label list 1 for the number zero elsewhere """
    return (y == number).astype(int)

""" Task D """

# Shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed[0:1000], y[0:1000], test_size=.2)


def model_digit(digit):

    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    #G = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    #plt.ion()
    plt.subplots(1,2)
    color=iter(plt.cm.rainbow(numpy.linspace(0,1,8)))

    for index,c in enumerate(C[0:1]):
        model = svm.SVC(C = c, kernel='linear', probability=True, gamma='auto', random_state=1)
        model = model.fit(X_train, label_list(y_train, digit))

        decision_score = model.decision_function(X_test)
        probability_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(label_list(y_test, digit), decision_score)
        precision, recall, _ = precision_recall_curve(label_list(y_test, digit), probability_score)

        accuracy = model.score(X_test, label_list(y_test, digit))

        print "C =", c, "and the accracy is :", accuracy
        print confusion_matrix(label_list(y_test, digit), model.predict(X_test))
        print classification_report(label_list(y_test, digit), model.predict(X_test))

        clr = next(color)
        lw = 2

        plt.subplot(1,2,1)
        plt.plot(fpr, tpr, color=clr, lw=lw, label='C =' + str(c))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, lw=lw, color=clr, label='C =' + str(c))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        #plt.pause(2.0)
        plt.draw()

    plt.show()

    return model

svm_digit_models = {}
for digit in range(0,1):
    svm_digit_models[digit] = model_digit(digit)


""" Testing """

DIGITS_DATASET = pandas.read_csv(DIGITS_PATH + 'test.csv')

TEST = DIGITS_DATASET.values.astype(numpy.uint8)
X_PCA = pca.fit_transform(TEST)

def taskd(X, X_PCA, models):
    # Test a few images
    for i in range(0,7):
        index = random.randint(0,1000)
        cv2.imshow("image", X[index].reshape(28,28)), cv2.waitKey(0)
        print X[index].shape
        X_reduced = X_PCA[index]
        prob_list = []
        for num in range(0, 10):
            prob_list.append(models[num].predict_proba(X_reduced)[0,1])
        print prob_list
        print 'Digit is :', prob_list.index(max(prob_list))

#taskd(TEST, X_PCA, svm_digit_models)


# Linear
# Any c
# G - not required for linear
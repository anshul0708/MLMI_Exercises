from __future__ import division

import os
import numpy
import pandas
import cv2
import random
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize

# Path to dataset
DIGITS_PATH = os.getcwd() + '/03-digits-dataset/'

""" Task A """

# Read train data into numpy array
DIGITS_DATASET = pandas.read_csv(DIGITS_PATH + 'train.csv')

X = DIGITS_DATASET.ix[:,1:].values.astype(numpy.uint8)
y = DIGITS_DATASET['label'].values.astype(numpy.uint8)

y = label_binarize(y, classes=range(0,10))

""" Task B """

pca = PCA(n_components=0.8)
X_transformed = pca.fit_transform(X)

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed[0:300], y[0:300], test_size=.2)


""" Create a new list for specific digit """

model = OneVsRestClassifier(svm.SVC(C = 1.0, kernel='linear', probability=True, gamma='auto'), n_jobs=-1)
model_score = model.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
thres = dict()
roc_auc = dict()
for i in range(0,10):
    fpr[i], tpr[i], thres[i] = roc_curve(y_test[:, i], model_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print fpr[2]

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

"""
def digit_test(models, X):
    #Test digit with every model
    prediction = []
    for number in range(0, 10):
        prediction.append(models[number].predict(X))
    return prediction

print digit_test(svm_models, X_transformed[20])

"""
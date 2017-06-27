from __future__ import division

import os
import pandas
import numpy
import warnings

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Ignore some warnings
warnings.filterwarnings("ignore")

PATH = os.getcwd() + '/04_RandomForests/'

# Import data into pandas data frames
TUBERCULOSIS_DATA = pandas.read_csv(PATH + 'TuberculosisData.csv')

# Create X and y
X = TUBERCULOSIS_DATA.ix[:,:-1].values
y = TUBERCULOSIS_DATA['Class'].values

# Create 5 fold dataset
kf = KFold(n_splits=5, shuffle=True, random_state=5)

random_forest_model = RandomForestClassifier(n_estimators=20, max_depth=4, warm_start=True)
logistic_regression_model = linear_model.LogisticRegression(C=1e6, solver='liblinear', fit_intercept=True, intercept_scaling=1e3, warm_start=True)
svm_model = svm.SVC(kernel='rbf', C=1.0, gamma=1.0)


# Split data into 5 folds
for train_index, test_index in kf.split(X):
    # Create the testing and the training fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #random_forest_model.set_params(n_estimators=25)
    random_forest_model.fit(X_train, y_train)
    logistic_regression_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    print '\n\nRandom Forest', random_forest_model.score(X_test, y_test)
    print 'LR', logistic_regression_model.score(X_test, y_test)
    print 'SVC', svm_model.score(X_test, y_test)


def measures(model, X, y):
    """ Calculate evaluation measures """

    # Predict labels
    predicted_labels = model.predict(X)

    # Initialize variables to calculate performance measures
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for (yp,yt) in zip(predicted_labels, y):
        if yp == yt:
            if yp == 2:
                TP += 1
            else:
                TN += 1
        else:
            if yp == 2:
                FP += 1
            else:
                FN += 1

    # Take class 2 as positive class
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print "\n\nClassification report for classifier", model
    print '\nAccuracy :', accuracy, '\nSensitivity :', sensitivity, '\nSpecificity :', specificity
    print confusion_matrix(y, predicted_labels)
    #print("Classification report for classifier %s:\n%s\n" % (model, metrics.classification_report(y, predicted_labels)))

measures(random_forest_model, X, y)
measures(logistic_regression_model, X, y)
measures(svm_model, X, y)

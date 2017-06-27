from __future__ import division

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy

from sklearn import svm

# Path to dataset
TWOMOONS_PATH = os.getcwd() + '/twomoons/'

""" Task A """

# Read train data into numpy array
X_twomoons = numpy.genfromtxt(TWOMOONS_PATH + 'xtrain.csv', delimiter=',')
y_twomoons = numpy.genfromtxt(TWOMOONS_PATH + 'ytrain.csv', delimiter=',').astype(int)

def taska():
    """ Plot data with different labels with different colors """
    plt.scatter(X_twomoons[:, 0], X_twomoons[:, 1], c=y_twomoons, cmap=plt.cm.coolwarm)
    plt.show()

taska()

""" Task B """

def svm_fit(X, y, kernel='rbf', c=1.0):
    """ Initialize a model, calculate measures on training data"""
    svm_model = svm.SVC(kernel=kernel, C=c)
    svm_model = svm_model.fit(X, y)
    predicted_labels = svm_model.predict(X)

    # Initialize variables to calculate performance measures
    T1 = 0
    T2 = 0
    F1 = 0
    F2 = 0

    for (yp,yt) in zip(predicted_labels, y):
        if yp == yt:
            if yp == 1:
                T1 += 1
            else:
                T2 += 1
        else:
            if yp == 1:
                F1 += 1
            else:
                F2 += 1

    # Take class 2 as positive class
    accuracy = (T1 + T2) / (T1 + T2 + F1 + F2)
    recall = T2 / (T2 + F1)
    fpr = F2 / (F2 + T1)
    precision = T2 / (T2 + F2)

    # Add performance measures into a list
    performance = [accuracy, recall, fpr, precision]

    return svm_model, predicted_labels, performance


""" Task C """

# Read train data into numpy array
Xtest_twomoons = numpy.genfromtxt(TWOMOONS_PATH + 'xtest.csv', delimiter=',')
ytest_twomoons = numpy.genfromtxt(TWOMOONS_PATH + 'ytest.csv', delimiter=',').astype(int)

def svm_predict(svm_model, X, y):
    """ Evaluate a model, calculate measures on testing data"""
    predicted_labels = svm_model.predict(X)

    # Initialize variables to calculate performance measures
    T1 = 0
    T2 = 0
    F1 = 0
    F2 = 0

    for (yp,yt) in zip(predicted_labels, y):
        if yp == yt:
            if yp == 1:
                T1 += 1
            else:
                T2 += 1
        else:
            if yp == 1:
                F1 += 1
            else:
                F2 += 1

    # Take class 2 as positive class
    accuracy = (T1 + T2) / (T1 + T2 + F1 + F2)
    recall = T2 / (T2 + F1)
    fpr = F2 / (F2 + T1)
    precision = T2 / (T2 + F2)

    # Add performance measures into a list
    performance = [accuracy, recall, fpr, precision]

    return predicted_labels, performance

""" Task D """

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

def training_plot(svm_data, X, y):
    """ Plot the graphs for task D """

    #Create sub plots
    plt.subplots(2,4)

    # Plot graph for each data
    for model in range(0,len(svm_data)):
        # Select sub plot
        plt.subplot(2,4,model+1)
        # create a mesh to plot in
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
        Z = svm_data[model][0].predict(numpy.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.title('Kernel : ' + svm_data[model][0].kernel +' , C = '+ str(C[model]))
        #Plot the graph with first the contour
        plt.contourf(xx, yy, Z, cmap= plt.cm.coolwarm, alpha=0.8)
        #Plot the points of training data
        plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm)
    plt.show()


def accuracy_plot(svm_data):
    """ Plot graph for training and testing accuracy change with c """
    training_accuracy = [model[2][0] for model in svm_data]
    plt.plot(C, training_accuracy, color='Red', label="Training")
    testing_accuracy = [model[4][0] for model in svm_data]
    plt.plot(C, testing_accuracy, color='Blue', label="Testing")
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.show()

# Compute models, predictions and performance for different C values in Linear model
linear_data = []
for c in C:
    svm_linear, predict_training, performance_training = svm_fit(X_twomoons, y_twomoons, kernel='linear', c=c)
    predict_testing, performance_testing = svm_predict(svm_linear, Xtest_twomoons, ytest_twomoons)
    linear_data.append([svm_linear, predict_training, performance_training, predict_testing, performance_testing])

# Compute models, predictions and performance for different C values
rbf_data = []
for c in C:
    svm_linear, predict_training, performance_training = svm_fit(X_twomoons, y_twomoons, kernel='rbf', c=c)
    predict_testing, performance_testing = svm_predict(svm_linear, Xtest_twomoons,ytest_twomoons)
    rbf_data.append([svm_linear, predict_training, performance_training, predict_testing, performance_testing])

# Linear
training_plot(linear_data, X_twomoons, y_twomoons)
accuracy_plot(linear_data)

# RBF
training_plot(rbf_data, X_twomoons, y_twomoons)
accuracy_plot(rbf_data)



""" Task E """

def color_list(y):
    """ Convert class labels to color """
    for (index,label) in zip(range(0, len(y)), y):
        if label == 1:
            y[index] = 'b'
        else:
            y[index] = 'r'
    return y

def C_animated_plot(model, X, y, Xtest, ytest):
    """ Print animated plot for different C values """
    for index,c in enumerate(C):
        plt.ion()

        plt.scatter(X[:, 0], X[:, 1], marker='o',cmap=plt.cm.coolwarm, facecolors='none', edgecolors=color_list(y.tolist()), label='Truth Training')
        plt.title("C: " + str(c))
        plt.legend()
        plt.pause(2.0)
        #plt.cla()

        plt.scatter(X[:, 0], X[:, 1], marker='+', cmap=plt.cm.coolwarm, c=color_list(model[index][1].tolist()), label='Prediction Training')
        plt.title("C: " + str(c))
        plt.legend()
        plt.pause(2.0)
        #plt.cla()

        plt.scatter(Xtest[:, 0], Xtest[:, 1], marker='s',cmap=plt.cm.coolwarm, facecolors='none', edgecolors=color_list(ytest.tolist()), label='Truth Testing')
        plt.title("C: " + str(c))
        plt.pause(2.0)
        plt.legend()
        #plt.cla()

        plt.scatter(Xtest[:, 0], Xtest[:, 1], marker='+', cmap=plt.cm.coolwarm, c=color_list(model[index][3].tolist()), label='Prediction Testing')
        plt.title("C: " + str(c))
        plt.legend()
        plt.pause(5.0)
        plt.cla()

        plt.draw()

C_animated_plot(rbf_data, X_twomoons, y_twomoons, Xtest_twomoons, ytest_twomoons)

"""
def plot_truth_pointwise(X, color, marker):
    plt.ion()
    for (x,c) in zip(X,color):
        plt.scatter(x[0], x[1], marker=marker, cmap=cm_bright, facecolors='none', edgecolors=c)
        plt.pause(1e-9)
    plt.draw()
    while True:
        plt.pause(0.05)

def plot_predicition_pointwise(ims, X, color, marker):
    for (x,c) in zip(X,color):
        im = plt.scatter(X[:, 0], X[:, 1], marker=marker, cmap=plt.cm.coolwarm, c=c)
        ims.append([im])
        print "Added 2"
    return ims


def animated_plot(svm_model, X, y, Xtest, ytest):
    #Draw a animated plot for different C values for training and testing data
    fig = plt.figure()

    ims = []
    for model in svm_model:
        plt.title("C: " + str(c))
        ims = plot_truth_pointwise(ims, X, color_list(y.tolist()), 'o')
        ims = plot_predicition_pointwise(ims, X, color_list(model[1].tolist()), '+')
        ims = plot_truth_pointwise(ims, Xtest, color_list(ytest.tolist()), 's')
        ims = plot_predicition_pointwise(ims, Xtest, color_list(model[1].tolist()), '+')
        #plt.cla()

    print len(ims)

    ani = animation.ArtistAnimation(fig, ims, interval=17, blit=True, repeat_delay=2000)
    plt.show()

#animated_plot(rbf_data[0:2], X_twomoons, y_twomoons, Xtest_twomoons, ytest_twomoons)
#plot_truth_pointwise(X_twomoons, color_list(y_twomoons.tolist()), 'o')


 
    ims = []
    for c in C:
        plt.title("C: " + str(c))
        im = plt.scatter(X[:, 0], X[:, 1], marker='o',cmap=plt.cm.coolwarm, facecolors='none', edgecolors=color_list(y.tolist()), label='Truth Training')
        ims.append([im])
        plt.scatter(X[:, 0], X[:, 1], marker='+', cmap=plt.cm.coolwarm, c=color_list(svm_model[0][1].tolist()), label='Prediction Training')
        ims.append([im])
        plt.scatter(Xtest[:, 0], Xtest[:, 1], marker='s', cmap=plt.cm.coolwarm, facecolors='none', edgecolors=color_list(ytest.tolist()), label='Truth Testing')
        ims.append([im])
        plt.scatter(Xtest[:, 0], Xtest[:, 1], marker='+', cmap=plt.cm.coolwarm, c=color_list(svm_model[0][3].tolist()), label='Prediction Testing')
        ims.append([im])
        #plt.cla()
"""
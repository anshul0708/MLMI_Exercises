from __future__ import division

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import os
import pandas
import numpy

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

PATH = os.getcwd() + '/04_RandomForests/'

# Import data into pandas data frames
TWIST_DATA = pandas.read_csv(PATH + 'TwistData.csv')
SPIRAL_DATA = pandas.read_csv(PATH + 'SpiralData.csv')

# Split into training and testing
TWIST_TRAIN, TWIST_TEST = train_test_split( TWIST_DATA, test_size = 0.2)
SPIRAL_TRAIN, SPIRAL_TEST = train_test_split( SPIRAL_DATA, test_size = 0.2)

""" Task B """

twist_models = {}
for num_tree in range(10,51,5):
    model = RandomForestClassifier(n_estimators=num_tree)
    twist_models[num_tree] = model.fit(TWIST_TRAIN.ix[:,:-1],TWIST_TRAIN['Class'])

spiral_models = {}
for num_tree in range(10,51,5):
    model = RandomForestClassifier(n_estimators=num_tree)
    spiral_models[num_tree] = model.fit(SPIRAL_TRAIN.ix[:,:-1], SPIRAL_TRAIN['Class'])

def random_forest_plot(random_forest_models, X_train, y_train, X_test, y_test, color_map, subplot_row, subplot_column):
    """ Plot the graphs for task B """

    #Create sub plots
    plt.subplots(subplot_row, subplot_column)

    # Subplot variable
    i = 1

    # Plot graph for each data
    for model in range(10,51,5):
        # Select sub plot
        plt.subplot(subplot_row, subplot_column, i)
        i += 1

        # Model score on test data
        score = random_forest_models[model].score(X_test, y_test)

        # create a mesh to plot in
        h = 0.01
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
        Z = random_forest_models[model].predict(numpy.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.title('No of tress : ' + str(model))
        #Plot the graph with first the contour
        plt.contourf(xx, yy, Z, cmap=color_map, alpha=0.5)
        #Plot the points of training data
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=color_map, edgecolors='k')
        #Plot the points of testing data with lighter shade
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=color_map, edgecolors='k', alpha=0.5)

        # Add model score
        score = random_forest_models[model].score(X_test, y_test)
        plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=10, horizontalalignment='right')

        # Model Limits
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
    plt.show()


# Colors for different classes
cm_3_class = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
random_forest_plot(twist_models, TWIST_TRAIN.ix[:,:-1].values, TWIST_TRAIN['Class'].values, TWIST_TEST.ix[:,:-1].values, TWIST_TEST['Class'].values, color_map=cm_3_class, subplot_row=3, subplot_column=3)
random_forest_plot(spiral_models, SPIRAL_TRAIN.ix[:,:-1].values, SPIRAL_TRAIN['Class'].values, SPIRAL_TEST.ix[:,:-1].values, SPIRAL_TEST['Class'].values, color_map=cm_3_class, subplot_row=3, subplot_column=3)

""" Task C """

def random_forest_depth_plot(random_forest_models, X_train, y_train, X_test, y_test, color_map, subplot_row, subplot_column):
    """ Plot the graphs for task B """

    #Create sub plots
    plt.subplots(subplot_row, subplot_column)

    # Subplot variable
    i = 1

    # Plot graph for each data
    for model in range(2,9):
        # Select sub plot
        plt.subplot(subplot_row, subplot_column, i)
        i += 1

        # Model score on test data
        score = random_forest_models[model].score(X_test, y_test)

        # create a mesh to plot in
        h = 0.01
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
        Z = random_forest_models[model].predict(numpy.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.title('Depth : ' + str(model))
        #Plot the graph with first the contour
        plt.contourf(xx, yy, Z, cmap=color_map, alpha=0.5)
        #Plot the points of training data
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=color_map, edgecolors='k')
        #Plot the points of testing data with lighter shade
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=color_map, edgecolors='k', alpha=0.5)

        # Add model score
        score = random_forest_models[model].score(X_test, y_test)
        plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=10, horizontalalignment='right')

        # Model Limits
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
    plt.show()

twist_depth_models = {}
for depth in range(2,9):
    model = RandomForestClassifier(n_estimators=10, max_depth=depth)
    twist_depth_models[depth] = model.fit(TWIST_TRAIN.ix[:,:-1],TWIST_TRAIN['Class'])

spiral_depth_models = {}
for depth in range(2,9):
    model = RandomForestClassifier(n_estimators=10, max_depth=depth)
    spiral_depth_models[depth] = model.fit(SPIRAL_TRAIN.ix[:,:-1], SPIRAL_TRAIN['Class'])

# Plots
random_forest_depth_plot(twist_depth_models, TWIST_TRAIN.ix[:,:-1].values, TWIST_TRAIN['Class'].values, TWIST_TEST.ix[:,:-1].values, TWIST_TEST['Class'].values, color_map=cm_3_class, subplot_row=2, subplot_column=4)
random_forest_depth_plot(spiral_depth_models, SPIRAL_TRAIN.ix[:,:-1].values, SPIRAL_TRAIN['Class'].values, SPIRAL_TEST.ix[:,:-1].values, SPIRAL_TEST['Class'].values, color_map=cm_3_class, subplot_row=2, subplot_column=4)

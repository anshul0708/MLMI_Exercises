import os
import cv2
import random
import scipy.io as sio
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

PATH = os.getcwd() + '/DRIVEdata.mat'
mat = sio.loadmat(PATH)
labels = mat['Labels']
images = mat['DataMatrix']

"""
for i in range(0, 4):
    index = random.randint(0, 1000)
    cv2.imshow("image", images[index].astype(numpy.uint8).reshape(25, 25)), cv2.waitKey(0)

print labels
"""

def load_data():
    X_train, X_test =train_test_split(images, test_size=0.2, random_state=42)
    X_train = X_train.astype('float32') / 255.
    # X_test = X_test.astype('float32') / 255.
    #X_train = normalize(X_train.astype('float32'))
    #X_test = normalize(X_test.astype('float32'))
    return X_train, X_test

def visualize_weights(W1, panel_shape, tile_size):
    # for coef, ax in zip(W1[0].T, axes.ravel()):
    #     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    # ax.set_xticks(())
    # ax.set_yticks(())
    print 'vizz'


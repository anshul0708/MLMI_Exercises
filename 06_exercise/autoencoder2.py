# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils

# Parameters
learning_rate = 1e-2
training_epochs = 200
batch_size = 100
display_step = 10


examples_to_show = 10
decay_steps = 100
decay_rate = 0.9

# Network Parameters
n_hidden_1 = 50 # 1st layer num features
std = 0.2
# n_hidden_2 = 125 # 2nd layer num features
n_input = 625 # MNIST data input (img shape: 25*25)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name='encoder_h1'),
    # 'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    # 'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input]),name='decoder_h1'),
    # 'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    # 'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    # 'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),    
    # 'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
    #                                biases['encoder_b2']))
    return layer_1


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
    #                                biases['decoder_b2']))
    return layer_1


# Construct model
# noise = tf.random_normal(shape=tf.shape(X), mean=0.0, stddev=std, dtype=tf.float32)
X_new = X  

encoder_op = encoder(X_new)

decoder_op = decoder(encoder_op)

global_steps = 0
losses =[]
# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# decayed_lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    train, test = utils.load_data()
    sess.run(init)
    total_batch = int(len(train)/batch_size)
    writer = tf.summary.FileWriter('/tmp/autolog', sess.graph)
    # Training cycle
    for epoch in range(training_epochs):
        global_steps = epoch
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = train[i*batch_size : (i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            losses.append(c)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
            

    print("Optimization Finished!")
    plt.plot(losses)
    plt.show()
    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: test[:examples_to_show]})
    
    cost = sess.run(cost, feed_dict={X: test})
    print(cost)
    # # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(test[i], (25, 25)))
        a[1][i].imshow(np.reshape(encode_decode[i], (25, 25)))
    
    weight1 = sess.run(weights['encoder_h1'])
    weight1 = weight1.T
    # hinton(weight1)
    vmin = weight1[0].min()
    vmax = weight1[0].max()
    w_plot, arange = plt.subplots(5, 10, figsize=(10, 10))
    new_wt = np.zeros((50,25,25))
    for i in range(len(weight1)):
        new_wt[i] = np.reshape(weight1[i],(-1, 25))
    weight1 = new_wt

    index = 0
    for i in range(10):
        if index < len(weight1):
            for j in range(0, 10):                
                arange[i][j].imshow(weight1[index],cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
                index += 1
    weight1 = sess.run(weights['decoder_h1'])
    vmin = weight1[0].min()
    vmax = weight1[0].max()
    w_plot2, arange2 = plt.subplots(5, 10, figsize=(10, 10))
    new_wt = np.zeros((50,25,25))
    for i in range(len(weight1)):
        new_wt[i] = np.reshape(weight1[i],(-1, 25))
    weight1 = new_wt

    index = 0
    for i in range(10):
        if index < len(weight1):
            for j in range(0, 10):                
                arange2[i][j].imshow(weight1[index],cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
                index += 1

    f.show()
    w_plot.show()
    w_plot2.show()
    
    plt.draw()
    plt.waitforbuttonpress()
    
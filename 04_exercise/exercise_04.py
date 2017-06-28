'''Softmax-Classifier for CIFAR-10'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import data_helpers

beginTime = time.time()

# Parameter definitions
batch_size = 50
learning_rate = 0.05
max_steps = 1000


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)




# Uncommenting this line removes randomness
# You'll get exactly the same result on each run
# np.random.seed(1)

# Prepare data
data_sets = data_helpers.load_data()

# -----------------------------------------------------------------------------
# Prepare the TensorFlow graph
# (We're only defining the graph here, no actual calculations taking place)
# -----------------------------------------------------------------------------

# Define input placeholders
# images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])
# labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# # Define variables (these are the values we want to optimize)
# weights = tf.Variable(tf.zeros([3072, 10]))
# biases = tf.Variable(tf.zeros([10]))


images_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])
keep_prob = tf.placeholder(tf.float32)

# images_placeholder = tf.reshape(x, [-1, 32, 32, 3])

#-----Layer1:Conv-----#

#-----Layer1:Conv-----#
with tf.name_scope('Layer1'):
    W_conv2 = weight_variable([3, 3, 3, 32])
    b_conv2 = bias_variable([32])
    with tf.name_scope('weights'):
        variable_summaries(W_conv2)
    
    with tf.name_scope('biases'):
        variable_summaries(b_conv2)

    h_conv2 = tf.nn.relu(conv2d(images_placeholder, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

#-----Layer2:Conv-----#
with tf.name_scope('Layer2'):
    W_conv3 = weight_variable([3, 3, 32, 64])
    b_conv3 = bias_variable([64])
    with tf.name_scope('weights'):
        variable_summaries(W_conv3)
    
    with tf.name_scope('biases'):
        variable_summaries(b_conv3)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

#-----Layer3:Conv-----#
with tf.name_scope('Layer3'):
    W_conv4 = weight_variable([3, 3, 64, 128])
    b_conv4 = bias_variable([128])
    with tf.name_scope('weights'):
        variable_summaries(W_conv4)
    
    with tf.name_scope('biases'):
        variable_summaries(b_conv4)

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

#-----Layer4:FC1-------3
with tf.name_scope('Layer4'):
    W_fc1 = weight_variable([4 * 4 * 128, 2048])
    b_fc1 = bias_variable([2048])

    with tf.name_scope('weights'):
        variable_summaries(W_fc1)
    
    with tf.name_scope('biases'):
        variable_summaries(b_fc1)

    h_pool2_flat = tf.reshape(h_pool4, [-1, 4 * 4 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#-----Layer5:FC2-------#
with tf.name_scope('Layer5'):
    W_fc2 = weight_variable([2048, 2])
    b_fc2 = bias_variable([2])

    with tf.name_scope('weights'):
        variable_summaries(W_fc2)
    
    with tf.name_scope('biases'):
        variable_summaries(b_fc2)

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define the classifier's result
logits = y_conv

# Define the loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
  labels=labels_placeholder))

# Define the training operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------

with tf.Session() as sess:
  # Initialize variables
  sess.run(tf.global_variables_initializer())

  # Repeat max_steps times
  for i in range(max_steps):

    # Generate input data batch
    indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
    images_batch = data_sets['images_train'][indices]
    labels_batch = data_sets['labels_train'][indices]
    # Periodically print out the model's current accuracy
    if i % 100 == 0:
      train_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: images_batch, labels_placeholder: labels_batch, keep_prob: 0.5})
      print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

    # Perform a single training step
    sess.run(train_step, feed_dict={images_placeholder: images_batch,
      labels_placeholder: labels_batch, keep_prob: 0.5})

  # After finishing the training, evaluate on the test set
  test_accuracy = sess.run(accuracy, feed_dict={
    images_placeholder: data_sets['images_test'],
    labels_placeholder: data_sets['labels_test'], keep_prob: 0.5})
  print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))

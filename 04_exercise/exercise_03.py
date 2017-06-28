"""
Solution for Exercise 03
"""

import tensorflow as tf

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

x = tf.placeholder(tf.float32, shape=[1, 64, 64, 3])

x_image = tf.reshape(x, [-1, 64, 64, 3])

#-----Layer1:Conv-----#
with tf.name_scope('Layer1'):
    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])
    with tf.name_scope('weights'):
        variable_summaries(W_conv1)
    
    with tf.name_scope('biases'):
        variable_summaries(b_conv1)

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

#-----Layer2:Conv-----#
with tf.name_scope('Layer2'):
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    with tf.name_scope('weights'):
        variable_summaries(W_conv2)
    
    with tf.name_scope('biases'):
        variable_summaries(b_conv2)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

#-----Layer3:Conv-----#
with tf.name_scope('Layer3'):
    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    with tf.name_scope('weights'):
        variable_summaries(W_conv3)
    
    with tf.name_scope('biases'):
        variable_summaries(b_conv3)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

#-----Layer4:Conv-----#
with tf.name_scope('Layer4'):
    W_conv4 = weight_variable([3, 3, 128, 256])
    b_conv4 = bias_variable([256])
    with tf.name_scope('weights'):
        variable_summaries(W_conv4)
    
    with tf.name_scope('biases'):
        variable_summaries(b_conv4)

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

#-----Layer5:FC1-------3
with tf.name_scope('Layer5'):
    W_fc1 = weight_variable([4 * 4 * 256, 2048])
    b_fc1 = bias_variable([2048])

    with tf.name_scope('weights'):
        variable_summaries(W_fc1)
    
    with tf.name_scope('biases'):
        variable_summaries(b_fc1)

    h_pool2_flat = tf.reshape(h_pool4, [-1, 4 * 4 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#-----Layer6:FC2-------#
with tf.name_scope('Layer6'):
    W_fc2 = weight_variable([2048, 2])
    b_fc2 = bias_variable([2])

    with tf.name_scope('weights'):
        variable_summaries(W_fc2)
    
    with tf.name_scope('biases'):
        variable_summaries(b_fc2)

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.Session() as sess:
    tf.global_variables_initializer()

    writer = tf.summary.FileWriter('/tmp/varlog', sess.graph)
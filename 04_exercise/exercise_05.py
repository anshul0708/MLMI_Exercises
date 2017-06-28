"""
Deep Learning Exercise: Exercise 05
"""
import tensorflow as tf

from inception_v3 import inception_v3
from inception_v3 import inception_v3_arg_scope

with tf.Graph().as_default():
    with tf.session as sess:
        images = tf.placeholder(tf.float32, shape=image_shape)
        with slim.arg_scope(inception_v3_arg_scope()):
                end_points = inception_v3(inputs=images)

        #Restore session
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        checkpoint_path = cur_dir + '/inception_v3.ckpt'
        restorer.restore(sess, checkpoint_path)
        sess.run(end_points, feed_dict{images: })

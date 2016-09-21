# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 23:58:29 2016

@author: bong
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:01:14 2016

@author: bong
"""

import tensorflow as tf
import imagenet_layer as layer
import numpy as np

CLASS_NUM = 1000

def preproc(image):
    
    
def inference(images, keep_prop):
    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d_layer(images, [5, 5, 3, 64], name=scope.name)
    
    with tf.variable_scope('res1') as scope:
        res1 = residual_layer(conv1, [3, 3, 64, 64], name=scope.name)

    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d_layer(res1, [3, 3, 64, 128], stride=2, name=scope.name)

    with tf.variable_scope('res2') as scope:
        res2 = residual_layer(conv2, [3, 3, 128, 128], name=scope.name)

    with tf.variable_scope('conv3') as scope:
        conv3 = conv2d_layer(res2, [3,3,128,256], stride=2, name=scope.name)
    
    with tf.variable_scope('res3') as scope:
        res3 = residual_layer(conv3, [3, 3, 256, 256], name=scope.name)

    with tf.variable_scope('avg_pool') as scope:
        avg_pool = tf.nn.avg_pool(res3, ksize=[1,8,8,1], strides=[1,1,1,1],
                                  padding='VALID', name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        reshaped = tf.reshape(avg_pool, [-1, 256])
        weights = weight_variable([256, LABEL_NUM], wd=0.0)
        biases = bias_variable([LABEL_NUM])
        softmax_linear = tf.add(tf.matmul(reshaped, weights), biases, name=scope.name)

    return softmax_linear

def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    
def train(total_loss, lr):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op
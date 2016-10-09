# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 23:58:29 2016

@author: bong
"""

import tensorflow as tf
import imagenet_layer as layer
import numpy as np
import time

def augment(images, BATCH_NUM):
    images=tf.convert_to_tensor(images, dtype=tf.uint8)
    
    res_dist=[]
    for i in xrange(BATCH_NUM):
        x=images[i,:,:,:]
        distort_x = tf.image.random_flip_left_right(x, seed=int(time.time()))
        distort_x = tf.image.per_image_whitening(distort_x)
        distort_x = tf.image.random_brightness(distort_x, max_delta=63)
        distort_x = tf.image.random_contrast(distort_x, lower=0.2, upper=1.8)
        res_dist.append(distort_x)
        
    return tf.convert_to_tensor(res_dist, dtype=tf.float32)
    
def inference(images, CLASS_NUM, is_training=False):
    conv1 = layer.conv2d_layer(images, [7,7,3,64], stride=2, activation_fn=None,name="conv1")
    batch1 = layer.bn_layer(conv1, is_training=is_training, name="batch1")
    max1 = layer.max_pool_layer(batch1, stride=2, name="max1")
    res1_1 = layer.residual_layer(max1, [3,3,64,64], name="res1_1")
    res1_2 = layer.residual_layer(res1_1, [3,3,64,64], name="res1_2")
    res1_3 = layer.residual_layer(res1_2, [3,3,64,64], name="res1_3")
    
    conv2 = layer.conv2d_layer(res1_3, [3,3,64,128], stride=2, activation_fn=None, name="conv2")
    batch2 = layer.bn_layer(conv2, is_training=is_training, name="batch2")
    res2_1 = layer.residual_layer(batch2, [3,3,128,128], name="res2_1")
    res2_2 = layer.residual_layer(res2_1, [3,3,128,128], name="res2_2")
    res2_3 = layer.residual_layer(res2_2, [3,3,128,128], name="res2_3")
    
    conv3 = layer.conv2d_layer(res2_3, [3,3,128,256], stride=2, activation_fn=None, name="conv3")
    batch3 = layer.bn_layer(conv3, is_training=is_training, name="batch3")
    res3_1 = layer.residual_layer(batch3, [3,3,256,256], name="res3_1")
    res3_2 = layer.residual_layer(res3_1, [3,3,256,256], name="res3_2")
    res3_3 = layer.residual_layer(res3_2, [3,3,256,256], name="res3_3")
    res3_4 = layer.residual_layer(res3_3, [3,3,256,256], name="res3_4")
    res3_5 = layer.residual_layer(res3_4, [3,3,256,256], name="res3_5")
    
    conv4 = layer.conv2d_layer(res3_5, [3,3,256,512], stride=2, activation_fn=None, name="conv4")
    batch4 = layer.bn_layer(conv4, is_training=is_training, name="batch4")    
    res4_1 = layer.residual_layer(batch4, [3,3,512,512], name="res4_1")
    res4_2 = layer.residual_layer(res4_1, [3,3,512,512], name="res4_2")
    
    with tf.variable_scope('avg_pool') as scope:
        avg_pool = tf.nn.avg_pool(res4_2, ksize=[1,7,7,1], strides=[1,1,1,1],
                                  padding='VALID', name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        reshaped = tf.reshape(avg_pool, [-1, 512])
        weights = layer.weight_variable([512, CLASS_NUM], wd=0.5)
        biases = layer.bias_variable([CLASS_NUM])
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
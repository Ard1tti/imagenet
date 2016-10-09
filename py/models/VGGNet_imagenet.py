# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 23:58:29 2016

@author: bong
"""

import tensorflow as tf
import imagenet_layer as layer
import numpy as np
import time
    
def inference(images, CLASS_NUM):
    conv1 = layer.conv2d_layer(images, [3,3,3,64], activation_fn=tf.nn.relu, name="conv1")
    max1 = layer.max_pool_layer(conv1, shape=[2,2,1], stride=2, name="max1")
    
    conv2 = layer.conv2d_layer(max1, [3,3,64,128], activation_fn=tf.nn.relu, name="conv2")
    max2 = layer.max_pool_layer(conv2, shape=[2,2,1], stride=2, name="max2")
    
    conv3 = layer.conv2d_layer(max2, [3,3,128,256], activation_fn=tf.nn.relu, name="conv3")
    conv3_1 = layer.conv2d_layer(conv3, [3,3,256,256], activation_fn=tf.nn.relu, name="conv3_1")
    max3 = layer.max_pool_layer(conv3_1, shape=[2,2,1], stride=2, name="max3")
    
    conv4 = layer.conv2d_layer(max3, [3,3,256,512], activation_fn=tf.nn.relu, name="conv4")
    conv4_1 = layer.conv2d_layer(conv4, [3,3,512,512], activation_fn=tf.nn.relu, name="conv4_1")
    max4 = layer.max_pool_layer(conv4_1, shape=[2,2,1], stride=2, name="max4")
    
    conv5 = layer.conv2d_layer(max4, [3,3,512,512], activation_fn=tf.nn.relu, name="conv5")
    conv5_1 = layer.conv2d_layer(conv5, [3,3,512,512], activation_fn=tf.nn.relu, name="conv5_1")
    max5 = layer.max_pool_layer(conv5_1, shape=[2,2,1], stride=2, name="max5")
    
    fc1 = layer.conv2d_layer(max5, [7,7,512,4096], activation_fn=tf.nn.relu,
                             padding="VALID", name="fc1")
    drop1 = tf.nn.dropout(fc1, 0.5, name="drop1")
    fc2 = layer.conv2d_layer(drop1, [1,1,4096,4096], activation_fn=tf.nn.relu,
                             padding="VALID", name="fc2")
    drop2 = tf.nn.dropout(fc2, 0.5, name="drop2")
    fc3 = layer.conv2d_layer(drop2, [1,1,4096,1000],
                             padding="VALID", name="softmax_linear")
    softmax_linear = tf.reshape(fc3, [-1,1000])
    return softmax_linear

def eval_once(images, CLASS_NUM):
    conv1 = layer.conv2d_layer(images, [3,3,3,64], activation_fn=tf.nn.relu, name="conv1")
    max1 = layer.max_pool_layer(conv1, shape=[2,2,1], stride=2, name="max1")
    
    conv2 = layer.conv2d_layer(max1, [3,3,64,128], activation_fn=tf.nn.relu, name="conv2")
    max2 = layer.max_pool_layer(conv2, shape=[2,2,1], stride=2, name="max2")
    
    conv3 = layer.conv2d_layer(max2, [3,3,128,256], activation_fn=tf.nn.relu, name="conv3")
    conv3_1 = layer.conv2d_layer(conv3, [3,3,256,256], activation_fn=tf.nn.relu, name="conv3_1")
    max3 = layer.max_pool_layer(conv3_1, shape=[2,2,1], stride=2, name="max3")
    
    conv4 = layer.conv2d_layer(max3, [3,3,256,512], activation_fn=tf.nn.relu, name="conv4")
    conv4_1 = layer.conv2d_layer(conv4, [3,3,512,512], activation_fn=tf.nn.relu, name="conv4_1")
    max4 = layer.max_pool_layer(conv4_1, shape=[2,2,1], stride=2, name="max4")
    
    conv5 = layer.conv2d_layer(max4, [3,3,512,512], activation_fn=tf.nn.relu, name="conv5")
    conv5_1 = layer.conv2d_layer(conv5, [3,3,512,512], activation_fn=tf.nn.relu, name="conv5_1")
    max5 = layer.max_pool_layer(conv5_1, shape=[2,2,1], stride=2, name="max5")
    
    fc1 = layer.conv2d_layer(max5, [7,7,512,4096], activation_fn=tf.nn.relu,
                             padding="VALID", name="fc1")
    fc2 = layer.conv2d_layer(fc1, [1,1,4096,4096], activation_fn=tf.nn.relu,
                             padding="VALID", name="fc2")
    fc3 = layer.conv2d_layer(fc2, [1,1,4096,1000],
                             padding="VALID", name="softmax_linear")
    softmax_linear = tf.reshape(fc3, [-1,1000])
    return tf.reshape(tf.reduce_mean(softmax_linear, reduction_indices=[0]),[1,1000])
    

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
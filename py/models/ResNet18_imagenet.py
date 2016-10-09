# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 23:58:29 2016

@author: bong
"""

import tensorflow as tf
import imagenet_layer as layer
import numpy as np
import time
    
def inference(images, CLASS_NUM, is_training=True):   
    conv1 = layer.conv2d_layer(images, [7,7,3,64], stride=2, activation_fn=None,name="conv1")
    batch1 = layer.bn_layer(conv1, is_training=is_training,
                            activation_fn = tf.nn.relu, name="batch1")
    max1 = layer.max_pool_layer(batch1, stride=2, name="max1")
    res1_1 = layer.residual_layer(max1, [3,3,64,64], is_training=is_training, name="res1_1")
    res1_2 = layer.residual_layer(res1_1, [3,3,64,64], is_training=is_training, name="res1_2")
    res1_3 = layer.residual_layer(res1_2, [3,3,64,64], is_training=is_training, name="res1_3")
    
    conv2 = layer.conv2d_layer(res1_3, [3,3,64,128], stride=2, activation_fn=None, name="conv2")
    batch2 = layer.bn_layer(conv2, is_training=is_training,
                            activation_fn = tf.nn.relu, name="batch2")
    res2_1 = layer.residual_layer(batch2, [3,3,128,128], is_training=is_training, name="res2_1")
    res2_2 = layer.residual_layer(res2_1, [3,3,128,128], is_training=is_training, name="res2_2")
    res2_3 = layer.residual_layer(res2_2, [3,3,128,128], is_training=is_training, name="res2_3")
    
    conv3 = layer.conv2d_layer(res2_3, [3,3,128,256], stride=2, activation_fn=None, name="conv3")
    batch3 = layer.bn_layer(conv3, is_training=is_training,
                            activation_fn = tf.nn.relu, name="batch3")
    res3_1 = layer.residual_layer(batch3, [3,3,256,256], is_training=is_training, name="res3_1")
    res3_2 = layer.residual_layer(res3_1, [3,3,256,256], is_training=is_training, name="res3_2")
    res3_3 = layer.residual_layer(res3_2, [3,3,256,256], is_training=is_training, name="res3_3")
    res3_4 = layer.residual_layer(res3_3, [3,3,256,256], is_training=is_training, name="res3_4")
    res3_5 = layer.residual_layer(res3_4, [3,3,256,256], is_training=is_training, name="res3_5")
    
    conv4 = layer.conv2d_layer(res3_5, [3,3,256,512], stride=2, activation_fn=None, name="conv4")
    batch4 = layer.bn_layer(conv4, is_training=is_training,
                            activation_fn = tf.nn.relu, name="batch4")    
    res4_1 = layer.residual_layer(batch4, [3,3,512,512], is_training=is_training, name="res4_1")
    res4_2 = layer.residual_layer(res4_1, [3,3,512,512], is_training=is_training, name="res4_2")
    
    avg_pool = tf.nn.avg_pool(res4_2, ksize=[1,7,7,1], strides=[1,1,1,1],
                              padding='VALID', name="avg_pool")
    softmax_linear = layer.conv2d_layer(avg_pool,[1,1,512,CLASS_NUM], name="softmax")
    return tf.reshape(softmax_linear, [-1,CLASS_NUM])

def eval_once(images, CLASS_NUM, is_training=False):
    conv1 = layer.conv2d_layer(images, [7,7,3,64], stride=2, activation_fn=None,name="conv1")
    batch1 = layer.bn_layer(conv1, is_training=is_training,
                            activation_fn = tf.nn.relu, name="batch1")
    max1 = layer.max_pool_layer(batch1, stride=2, name="max1")
    res1_1 = layer.residual_layer(max1, [3,3,64,64], is_training=is_training, name="res1_1")
    res1_2 = layer.residual_layer(res1_1, [3,3,64,64], is_training=is_training, name="res1_2")
    res1_3 = layer.residual_layer(res1_2, [3,3,64,64], is_training=is_training, name="res1_3")
    
    conv2 = layer.conv2d_layer(res1_3, [3,3,64,128], stride=2, activation_fn=None, name="conv2")
    batch2 = layer.bn_layer(conv2, is_training=is_training,
                            activation_fn = tf.nn.relu, name="batch2")
    res2_1 = layer.residual_layer(batch2, [3,3,128,128], is_training=is_training, name="res2_1")
    res2_2 = layer.residual_layer(res2_1, [3,3,128,128], is_training=is_training, name="res2_2")
    res2_3 = layer.residual_layer(res2_2, [3,3,128,128], is_training=is_training, name="res2_3")
    
    conv3 = layer.conv2d_layer(res2_3, [3,3,128,256], stride=2, activation_fn=None, name="conv3")
    batch3 = layer.bn_layer(conv3, is_training=is_training,
                            activation_fn = tf.nn.relu, name="batch3")
    res3_1 = layer.residual_layer(batch3, [3,3,256,256], is_training=is_training, name="res3_1")
    res3_2 = layer.residual_layer(res3_1, [3,3,256,256], is_training=is_training, name="res3_2")
    res3_3 = layer.residual_layer(res3_2, [3,3,256,256], is_training=is_training, name="res3_3")
    res3_4 = layer.residual_layer(res3_3, [3,3,256,256], is_training=is_training, name="res3_4")
    res3_5 = layer.residual_layer(res3_4, [3,3,256,256], is_training=is_training, name="res3_5")
    
    conv4 = layer.conv2d_layer(res3_5, [3,3,256,512], stride=2, activation_fn=None, name="conv4")
    batch4 = layer.bn_layer(conv4, is_training=is_training,
                            activation_fn = tf.nn.relu, name="batch4")    
    res4_1 = layer.residual_layer(batch4, [3,3,512,512], is_training=is_training, name="res4_1")
    res4_2 = layer.residual_layer(res4_1, [3,3,512,512], is_training=is_training, name="res4_2")
    
    avg_pool = tf.nn.avg_pool(res4_2, ksize=[1,7,7,1], strides=[1,1,1,1],
                              padding='VALID', name="avg_pool")
    softmax_linear = layer.conv2d_layer(avg_pool,[1,1,512,CLASS_NUM], name="softmax")
    softmax_linear = tf.reshape(softmax_linear, [-1,CLASS_NUM])
    return tf.reshape(tf.reduce_mean(softmax_linear, reduction_indices=[0]),[1,CLASS_NUM])

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
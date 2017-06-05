#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Create model
def conv_net(x, params):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, 64, 64, 3])

	# Convolution and max pooling(down-sampling) Layer
	conv11 = conv2d(x, params['W_conv11'], params['b_conv11'])
	conv12 = conv2d(conv11, params['W_conv12'], params['b_conv12'])
	conv13 = conv2d(conv12, params['W_conv13'], params['b_conv13'])
	conv13 = maxpool2d(conv13, k=2)

	# Convolution and max pooling(down-sampling) Layer
	conv21 = conv2d(conv13, params['W_conv21'], params['b_conv21'])
	conv22 = conv2d(conv21, params['W_conv22'], params['b_conv22'])
	conv23 = conv2d(conv22, params['W_conv23'], params['b_conv23'])
	conv23 = maxpool2d(conv23, k=2)

	# Convolution and max pooling(down-sampling) Layer
	conv31 = conv2d(conv23, params['W_conv31'], params['b_conv31'])
	conv32 = conv2d(conv31, params['W_conv32'], params['b_conv32'])
	conv33 = conv2d(conv32, params['W_conv33'], params['b_conv33'])
	conv33 = maxpool2d(conv33, k=2)

	# Convolution and max pooling(down-sampling) Layer
	# conv41 = conv2d(conv33, params['W_conv41'], params['b_conv41'])
	# conv42 = conv2d(conv41, params['W_conv42'], params['b_conv42'])
	# conv43 = conv2d(conv42, params['W_conv43'], params['b_conv43'])
	# conv43 = maxpool2d(conv43, k=2)

	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv33, [-1, params['W_fc1'].get_shape().as_list()[0]])

	# Fully connected layer and Apply Dropout
	fc1 = tf.add(tf.matmul(fc1, params['W_fc1']), params['b_fc1'])
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, params['dropout'])

	# Fully connected layer and Apply Dropout
	fc2 = tf.add(tf.matmul(fc1, params['W_fc2']), params['b_fc2'])
	fc2 = tf.nn.relu(fc2)
	fc2 = tf.nn.dropout(fc2, params['dropout'])

	# Output, class prediction
	out = tf.add(tf.matmul(fc2, params['W_out']), params['b_out'])
	# out = tf.nn.l2_normalize(out, 1, epsilon=1e-12, name=None)
	return out
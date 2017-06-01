#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os

# Import packages
import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from numpy import genfromtxt


from customhelpers.image_handler import Image_handler
from customhelpers.image_for_tf import Image_for_tf
from customhelpers.customhelpers import show_from_serialized_img
from model_body import conv_net
from params import params

kaggle_catdog_test = Image_for_tf(os.path.dirname(os.getcwd()) +"\\data\\Kaggle_catdog\\")
kaggle_catdog_test.import_data(["kaggle_catdog_train_64x64.pickle"])
kaggle_catdog_test.filter_classes([3,5])
kaggle_catdog_test.encode_onehot(zero_columns=False)
# kaggle_catdog_test.normalize_axis1()
kaggle_catdog_test.shuffle()

image_handler = Image_handler()
image_handler.open_image("saechi.png")
# image_handler.image.show()
image_handler.adjust_size((64,64))
image_handler.serialize_image(4096,4)
image_handler.reset_datatype()
saechi_ser = image_handler.arr_img

image_handler = Image_handler()
image_handler.open_image("molly.jpg")
# image_handler.image.show()
image_handler.adjust_size((64,64))
image_handler.serialize_image2()
image_handler.reset_datatype()
molly_ser = image_handler.arr_img


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Create model
def conv23_inter(x, params):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 64, 64, 3])

    # Convolution and max pooling(down-sampling) Layer
    conv11 = conv2d(x, params['W_conv11'], params['b_conv11'])
    # conv11 = tf.nn.dropout(conv11, params['d_conv'])
    conv12 = conv2d(conv11, params['W_conv12'], params['b_conv12'])
    conv13 = conv2d(conv12, params['W_conv13'], params['b_conv13'])
    conv13 = maxpool2d(conv13, k=2)

    # Convolution and max pooling(down-sampling) Layer
    conv21 = conv2d(conv13, params['W_conv21'], params['b_conv21'])
    conv22 = conv2d(conv21, params['W_conv22'], params['b_conv22'])
    conv23 = conv2d(conv22, params['W_conv23'], params['b_conv23'])
    # conv23 = maxpool2d(conv23, k=2)
    return conv22
# Define saver 
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    # Initialise the variables and run
    init = tf.global_variables_initializer()
    sess.run(init)
    
    with tf.device("/cpu:0"):
        # Restore saved model if any
        try:
            saver.restore(sess, ".\\model\\model.ckpt")
            print("Model restored")
        except tf.errors.NotFoundError:
            print("No saved model found")

        # for rec_ser in kaggle_catdog_test.data[:10,:-2]:
        #     rec_ser = rec_ser.astype(np.float32, copy=False)
        #     estm = conv_net(rec_ser, params)
        #     estm = estm.eval()[0]
        #     print(estm)
        #     if estm[0] > estm[1]:
        #         print([1,0])
        #     else:
        #         print([0,1])
        #     show_from_serialized_img(rec_ser, 3, (64, 64))

        conv23 = conv23_inter(molly_ser, params) # (1,16(x),16(y),64)
        conv23 = conv23.eval()[0] 
        conv23 = np.swapaxes(conv23, 0, 2) # (16(x),16(y),64)
        conv23 = np.swapaxes(conv23, 1, 2) # (64,16(x),16(y))

        for rec in conv23[0:25]:
            show_from_serialized_img(rec, 1, (32, 32))

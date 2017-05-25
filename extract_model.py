#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os

# Import 3rd party packages
import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import tensorflow as tf

# Import custom packages
from customhelpers.image_for_tf import Image_for_tf
from params import params
from model_body import conv_net

root = os.path.dirname(os.getcwd())
# Import data
kaggle_catdog_test = Image_for_tf(root +'\\data\\Kaggle_catdog\\kaggle_catdog_test_64x64.pickle')
kaggle_catdog_test.filter_classes([3,5])
kaggle_catdog_test.encode_onehot(6, zero_columns=False)
kaggle_catdog_test.shuffle()

im = Image.open(os.getcwd() + "\\publish\\saechi.jpg")
im = im.resize((64,64))
im_arr = np.asarray(im)
im_arr = im_arr.astype(np.float32, copy=False)

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

        for rec_ser in kaggle_catdog_test.data[:10,:-2]:
            rec_ser = rec_ser.astype(np.float32, copy=False)
            rec_ser = rec_ser/256
            estm = conv_net(rec_ser, params)
            estm = estm.eval()[0]
            print(estm)
            if estm[0] > estm[1]:
                print([1,0])
            else:
                print([0,1])

            rec = rec_ser.reshape((64,64,3))

            plt.imshow(rec)
            plt.show()
            print("++++++++++++++++++++")
            


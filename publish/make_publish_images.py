#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import sys, os
sys.path.insert(0, '..\\')

# Import 3rd party packages
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import tensorflow as tf

# Import custom packages
from model_body import conv2d, maxpool2d, conv_net
from params import params

root = os.path.dirname(os.getcwd())

im = Image.open(os.getcwd() + "\\saechi.jpg")
im = im.resize((64,64))
im_arr = np.asarray(im)
im_arr = im_arr.astype(np.float32, copy=False)

# Create model
def conv_net_inter(x, params):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, 64, 64, 3])

	# Convolution and max pooling(down-sampling) Layer
	conv11 = conv2d(x, params['W_conv11'], params['b_conv11'])
	conv12 = conv2d(conv11, params['W_conv12'], params['b_conv12'])
	conv13 = conv2d(conv12, params['W_conv13'], params['b_conv13'])
	conv13p = maxpool2d(conv13, k=2)

	# Convolution and max pooling(down-sampling) Layer
	conv21 = conv2d(conv13p, params['W_conv21'], params['b_conv21'])
	conv22 = conv2d(conv21, params['W_conv22'], params['b_conv22'])
	conv23 = conv2d(conv22, params['W_conv23'], params['b_conv23'])
	conv23p = maxpool2d(conv23, k=2)

	# Convolution and max pooling(down-sampling) Layer
	conv31 = conv2d(conv23p, params['W_conv31'], params['b_conv31'])
	conv32 = conv2d(conv31, params['W_conv32'], params['b_conv32'])
	conv33 = conv2d(conv32, params['W_conv33'], params['b_conv33'])
	conv33p = maxpool2d(conv33, k=2)

	# # Convolution and max pooling(down-sampling) Layer
	# conv41 = conv2d(conv33p, params['W_conv41'], params['b_conv41'])
	# conv42 = conv2d(conv41, params['W_conv42'], params['b_conv42'])
	# conv43 = conv2d(conv42, params['W_conv43'], params['b_conv43'])
	# conv43p = maxpool2d(conv43, k=2)

	return conv13

# Function to transpose the shape of tensor of intermediate convolution layers
def swap_inter(tnsr_input):
	ret = tnsr_input.eval()[0] 
	ret = np.swapaxes(ret, 0, 2) # (16(x),16(y),64)
	ret = np.swapaxes(ret, 1, 2) # (64,16(x),16(y))
	return ret

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
			saver.restore(sess, root + "\\model\\model.ckpt")
			print("Model restored")
		except tf.errors.NotFoundError:
			print("No saved model found")

		# Transpose the shape of tensor of intermediate convolution layers
		tnsr_input = conv_net_inter(im_arr, params) # (1, 16(x), 16(y), 64)
		input_inter = swap_inter(tnsr_input) # (64, 16(x), 16(y))


		# Plot for publish images
		fig, axes = plt.subplots(nrows=4, ncols=4)
		ax = axes.flatten()
		i = 0
		while i < 16:
			rec = input_inter[i+16]

			shape = rec.shape
			size_img = shape[0]*shape[1]
			img_rgb_tp = rec.reshape(size_img)

			img_rgb = img_rgb_tp.transpose()
			img_rgb = img_rgb.reshape(shape[0], shape[1])
			img_rgb = img_rgb.astype(np.float32, copy=False)
			img_rgb = img_rgb

			ax[i].imshow(img_rgb, cmap='gray')
			ax[i].get_xaxis().set_visible(False)
			ax[i].get_yaxis().set_visible(False)

			i += 1

		fig.tight_layout()
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.show()
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
from params import params

from customhelpers.image_handler import Image_handler
from customhelpers.image_for_tf import Image_for_tf
from customhelpers.customhelpers import show_from_serialized_img
from model_body import conv_net

# Import data
kaggle_catdog = Image_for_tf(os.path.dirname(os.getcwd()) +'\\data\\Kaggle_catdog\\')
kaggle_catdog.import_data(['kaggle_catdog_train_64x64.pickle'])
kaggle_catdog.filter_classes([3,5])
kaggle_catdog.encode_onehot(zero_columns=False)
# kaggle_catdog.normalize_axis1()
kaggle_catdog.shuffle()

kaggle_catdog_test = Image_for_tf(os.path.dirname(os.getcwd()) +'\\data\\Kaggle_catdog\\')
kaggle_catdog_test.import_data(['kaggle_catdog_test_64x64.pickle'])
kaggle_catdog_test.filter_classes([3,5])
kaggle_catdog_test.encode_onehot(zero_columns=False)
# kaggle_catdog_test.normalize_axis1()
kaggle_catdog_test.shuffle()

data_training = kaggle_catdog.data
data_test = kaggle_catdog_test.data[:128]

# BUILDING THE COMPUTATIONAL GRAPH
# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 256
display_step = 10

# Network Parameters
n_input = 4096*3 # 64*64*3
n_classes = 2 # cat or dog

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

data_saved = {'var_epoch_saved': tf.Variable(0)}

# Construct model
pred = conv_net(x, params)

# Define loss and optimiser
crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(crossEntropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# RUNNING THE COMPUTATIONAL GRAPH
# Define saver 
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	step = 1

	with tf.device('/cpu:0'):
		# Restore saved model if any
		try:
			saver.restore(sess, '.\\model\\model.ckpt')
			print('Model restored')
			epoch_saved = data_saved['var_epoch_saved'].eval()
		except tf.errors.NotFoundError:
			print('No saved model found')
			epoch_saved = 0
		except tf.errors.InvalidArgumentError:
			print('Model structure has change. Rebuild model')
			epoch_saved = 0

		# Training cycle
		print(epoch_saved)
		batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]
		for epoch in range(epoch_saved, epoch_saved + training_epochs):
			batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]
			batch_x = batch[:, :4096*3]
			batch_y = batch[:, 4096*3:]
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: params['dropout']})
			if epoch % display_step == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
				# print('Epoch ' + str(epoch) + ', Minibatch Loss= ' + '{:.6f}'.format(loss) + ', Training Accuracy= ' + '{:.5f}'.format(acc))
		
				# Validation
				acc_test = sess.run(accuracy, feed_dict={x: data_test[:,:4096*3], y: data_test[:,4096*3:], keep_prob: 1.})
				print('Epoch ' + str(epoch) + ', Minibatch Loss= ' + '{:.6f}'.format(loss) + ', Training Accuracy= ' + '{:.5f}'.format(acc) + ', Validation Accuracy= ' + '{:.5f}'.format(acc_test))

				# batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]
		print('Optimisation Finished!')

		# Save the variables
		epoch_new = epoch_saved + training_epochs
		sess.run(data_saved['var_epoch_saved'].assign(epoch_saved + training_epochs))
		print(data_saved['var_epoch_saved'].eval())
		save_path = saver.save(sess, '.\\model\\model.ckpt')
		print('Model saved in file: %s' % save_path)

		
input('PRESS ANY KEY TO QUIT')
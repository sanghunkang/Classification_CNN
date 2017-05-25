#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os

# Import 3rd party packages
import numpy as np
import tensorflow as tf

# Import custom packages
from customhelpers.image_for_tf import Image_for_tf
from params import params
from model_body import conv_net

root = os.path.dirname(os.getcwd())
# Import data
# kaggle_catdog = Image_for_tf(root +'\\data\\Kaggle_catdog\\kaggle_catdog_train_64x64.pickle')
# kaggle_catdog.filter_classes([3,5])
# kaggle_catdog.encode_onehot(6, zero_columns=False)
# kaggle_catdog.shuffle()
# filepaths = [
# 	root +'\\data\\Kaggle_catdog\\kaggle_catdog_train_128x128_1.pickle',
# 	# root +'\\data\\Kaggle_catdog\\kaggle_catdog_train_128x128_2.pickle',
# 	# root +'\\data\\Kaggle_catdog\\kaggle_catdog_train_128x128_3.pickle',
# 	# root +'\\data\\Kaggle_catdog\\kaggle_catdog_train_128x128_4.pickle',
# ]
kaggle_catdog1 = Image_for_tf([root +'\\data\\Kaggle_catdog\\kaggle_catdog_train_64x64.pickle'], isOnehot=True)
kaggle_catdog1.shuffle()

# kaggle_catdog2 = Image_for_tf([root +'\\data\\Kaggle_catdog\\kaggle_catdog_train_128x128_2.pickle'], isOnehot=True)
# kaggle_catdog2.shuffle()
# kaggle_catdog3 = Image_for_tf([root +'\\data\\Kaggle_catdog\\kaggle_catdog_train_128x128_3.pickle'], isOnehot=True)
# kaggle_catdog3.shuffle()
# kaggle_catdog4 = Image_for_tf([root +'\\data\\Kaggle_catdog\\kaggle_catdog_train_128x128_4.pickle'], isOnehot=True)
# kaggle_catdog4.shuffle()

kaggle_catdog_test = Image_for_tf([root +'\\data\\Kaggle_catdog\\kaggle_catdog_test_64x64.pickle'], isOnehot=True)
kaggle_catdog_test.shuffle()

data_training = np.concatenate([kaggle_catdog1.data])#, kaggle_catdog2.data, kaggle_catdog3.data, kaggle_catdog4.data])
data_test = kaggle_catdog_test.data
print(data_training.shape)
print(data_test.shape)

# BUILDING THE COMPUTATIONAL GRAPH
# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 64
display_step = 10

with tf.name_scope('hidden') as scope:
	params = params

# Network Parameters
n_input = 64*64*3 # 64*64*3
n_classes = 2 # cat or dog

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

data_saved = {'var_epoch_saved': tf.Variable(0)}

# Construct model
pred = conv_net(x, params)

# Define loss and optimiser
crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(crossEntropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

def feed_dict(train):
	"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
	if train:
		xs, ys = batch_x, batch_y
		k = params['dropout']
	else:
		batch_test = data_test[np.random.choice(data_test.shape[0], size=batch_size,  replace=True)]
		xs, ys =  batch_test[:,:64*64*3], batch_test[:,64*64*3:]
		k = 1.0
	return {x: xs, y: ys, keep_prob: k}

# RUN THE COMPUTATIONAL GRAPH
# Define saver 
merged = tf.summary.merge_all()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
	summaries_dir = '.\\logs'
	train_writer = tf.summary.FileWriter(summaries_dir + '\\train', sess.graph)
	test_writer = tf.summary.FileWriter(summaries_dir + '\\test')
	tf.global_variables_initializer().run()
	step = 1

	with tf.device('/gpu:0'):
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
		for epoch in range(epoch_saved, epoch_saved + training_epochs):
			batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]
			batch_x = batch[:, :64*64*3]
			batch_y = batch[:, 64*64*3:]
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: params['dropout']})
			if epoch % display_step == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
				print('Epoch ' + str(epoch) + ', Minibatch Loss= ' + '{:.6f}'.format(loss) + ', Training Accuracy= ' + '{:.5f}'.format(acc))# + ', Validation Accuracy= ' + '{:.5f}'.format(acc_test))

				# batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]

			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, epoch)
			
			print('Test Accuracy at step %s: %s' % (epoch, acc))

			summary, _ = sess.run([merged, optimizer], feed_dict=feed_dict(True))
			train_writer.add_summary(summary, epoch)
	
		print('Optimisation Finished!')

		# Save the variables
		epoch_new = epoch_saved + training_epochs
		sess.run(data_saved['var_epoch_saved'].assign(epoch_saved + training_epochs))
		print(data_saved['var_epoch_saved'].eval())
		save_path = saver.save(sess, '.\\model\\model.ckpt')
		print('Model saved in file: %s' % save_path)

# input('PRESS ANY KEY TO QUIT')
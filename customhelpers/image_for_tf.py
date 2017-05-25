#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os
import _pickle as cPickle
import pickle as pickle

# Import 3rd party packages
import numpy as np

class Image_for_tf():
	def __init__(self, filepaths, isOnehot=False):
		self.isOnehot = isOnehot

		with open(filepaths[0], "rb") as fo:
			self.data = cPickle.load(fo, encoding="bytes")
		if len(filepaths) > 1:
			for filepath in filepaths:
				with open(filepath, "rb") as fo:
					raw_data = cPickle.load(fo, encoding="bytes")
					self.data = np.concatenate([self.data, raw_data])

	def filter_classes(self, classes):
		"""
		classes			: list, 
		"""
		data_tmp = []
		for i in classes:
			data_tmp.append(self.data[self.data[:,-1] == i])
		self.data = np.concatenate(data_tmp)

	def encode_onehot(self, n_classes, zero_columns=True):
		"""
		n_classes		: int, Biggest number for label
		zero_columns	: boolean, Whether to shrink columns if only zeros appear
		"""
		if self.isOnehot == True:
			print("The sequence is already onehot encoded")
		elif self.isOnehot == False:
			# Determine shape of the output one-hot matrix
			seq = self.data[:,-1]
			seq = seq.astype(int, copy=False)
			n_rec = len(seq)

			# Create the output one-hot label matrix 
			mat = np.zeros((n_rec, n_classes))
			mat[np.arange(n_rec), seq] = 1

			# Omit all-zero columns if specified
			if zero_columns == False:
				ret = mat[:, np.apply_along_axis(np.count_nonzero, 0, mat) > 0]
			
			# Replace original labels with onehot-encoded labels 
			self.data = np.c_[self.data[:,:-1], ret]
			self.isOnehot = True

	def normalize_axis1(self):
		if self.isOnehot == True:
			data_X = self.data[:,:-2]
			len_seq = len(data_X)

			data_X1 = data_X[:,4096*0:4096*1]
			data_X2 = data_X[:,4096*1:4096*2]
			data_X3 = data_X[:,4096*2:4096*3]
			
			data_X1 -= np.mean(data_X1, axis=1).reshape(len_seq,1) # zero-center
			data_X1 /= np.std(data_X1, axis=1).reshape(len_seq,1) # normalize

			data_X2 -= np.mean(data_X2, axis=1).reshape(len_seq,1) # zero-center
			data_X2 /= np.std(data_X2, axis=1).reshape(len_seq,1) # normalize

			data_X3 -= np.mean(data_X3, axis=1).reshape(len_seq,1) # zero-center
			data_X3 /= np.std(data_X3, axis=1).reshape(len_seq,1) # normalize

			self.data = np.c_[data_X1, data_X2, data_X3, self.data[:,-2:]]

	def normalize_axis0(self):
		if self.isOnehot == True:
			data_X = self.data[:,:-2]
			
			data_X -= np.mean(data_X, axis=0) # zero-center
			data_X /= np.std(data_X, axis=0) # normalize

			self.data = np.c_[data_X, self.data[:,-2:]]


	def whiten(self):
		if self.isOnehot == True:
			data_X = self.data[:,:-2]
			
			data_X -= np.mean(data_X, axis = 0) # zero-center
			cov = np.dot(data_X.T, data_X) / data_X.shape[0] # compute the covariance matrix
			U,S,V = np.linalg.svd(cov) # compute the SVD factorization of the data covariance matrix
			data_Xrot = np.dot(data_X, U) # decorrelate the data
			data_Xwhite = data_Xrot / np.sqrt(S + 1e-5) # divide by the eigenvalues (which are square roots of the singular values)
			self.data = np.c_[data_X, self.data[:,:-2]]

	def shuffle(self):
		"""
		Shuffle the sequnce of the data
		"""
		np.random.shuffle(self.data)

	def get_batch(self, batchsize):
		"""
		Generate a random batch with input batchsize
		batchsize		: int,
		"""
		pass
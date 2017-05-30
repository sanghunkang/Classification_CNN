#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
import random as random

import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import PIL as pil
import PIL.Image as Image

# with open(".\\data\\data_batch_2", "rb") as fo:
#     raw_data = cPickle.load(fo, encoding="bytes")
#     data_input, data_output = raw_data[b'data'], raw_data[b'labels']

# data2 = np.c_[data_input, data_output]

class Image_handler():
	def __init__(self):
		self.image_open = None
		self.arr_img = None

	def open_image(self, path):
		self.image = pil.Image.open(path)
		self.arr_img = np.asarray(self.image)

	def adjust_size(self, shape):
		self.image = self.image.resize(shape)
		self.arr_img = np.asarray(self.image)

	def serialize_image(self, length, channel):
		self.arr_img = self.arr_img.reshape(1, length, channel)[0]
		self.arr_img = self.arr_img.transpose()
		self.arr_img = np.concatenate([self.arr_img[0],self.arr_img[1],self.arr_img[2]])

	def serialize_image2(self):
		self.arr_img = self.arr_img.reshape(1, 12288)[0]
	
	def reset_datatype(self, dtype=np.float32):
		self.arr_img = self.arr_img.astype(dtype, copy=False)



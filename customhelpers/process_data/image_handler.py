#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
import random as random

import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import PIL as pil
import PIL.Image as Image

def open_image(path_img):
	ret = Image.open(path_img)
	return ret

def crop_innersqr(img_open):
	size_img = img_open.size
	min_side = min(size_img)
	padding_h, padding_v = (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
	ret = img_open.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))
	return ret

def reshape_image(img_open, shape):
	ret = img_open.resize(shape)
	return ret

def serialize_image(img_open):
	img_serialized = np.asarray(img_open)
	ret = img_serialized.reshape(1, img_serialized.size)[0]
	return ret

def process_image(path_img, resolution):
	img_open = open_image(path_img)
	img_cropped = crop_innersqr(img_open)
	img_reshaped = reshape_image(img_cropped, resolution)
	ret = serialize_image(img_reshaped)		
	return ret


#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import PIL.Image as Image

# def open_image(path_img):
# 	return Image.open(path_img)

def crop_innersqr(img_open):
	size_img = img_open.size
	min_side = min(size_img)
	padding_h, padding_v = (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
	img_open.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))
	return img_open

def reshape_image(img_open, shape):
	ret = img_open.resize(shape) 
	return ret

def serialize_image(img_open):
	img_serialized = np.asarray(img_open)
	ret = img_serialized.reshape(1, img_serialized.size)[0]
	return ret

def process_image(path_img, resolution):
	img_open = Image.open(path_img)
	img_cropped = crop_innersqr(img_open)
	img_reshaped = img_cropped.resize(resolution) 
	# img_reshaped = reshape_image(img_cropped, resolution)
	return serialize_image(img_reshaped)
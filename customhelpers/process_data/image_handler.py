#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import PIL.Image as Image

# def open_image(path_img):
# 	return Image.open(path_img)
path_img = "C:\\dev\\lab_fda\\data\\Kaggle_catdog\\test\\cat.10007.jpg"
def crop_innersqr(img_open):
	size_img = img_open.size
	min_side = min(size_img)
	padding_h, padding_v = (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
	img_open.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))
	return img_open

def reshape_image(img_open, shape):
	ret = img_open.resize(shape) 
	return ret

def swap_format():
	pass

def serialize_image(img_open):
	img_serialized = np.asarray(img_open)
	ret = img_serialized.reshape(1, img_serialized.size)[0]
	return ret

def process_image(path_img, resolution, format_arr="HWC"):
	img_open = Image.open(path_img)
	img_cropped = crop_innersqr(img_open)
	img_reshaped = img_cropped.resize(resolution)
	arr_img = serialize_image(img_reshaped)
	# img_reshaped = reshape_image(img_cropped, resolution)
	if format_arr == "CHW":
		arr_img_reformat = reformat_image(arr)
		return arr_img_reformat
	elif format_arr == "HWC":
		return arr_img

def reformat_image(img_serialized):
	arr_reformat = np.append(img_serialized[0::3], img_serialized[1::3])
	arr_reformat = np.append(arr_reformat, img_serialized[2::3])
	return arr_reformat

# img_open = Image.open(path_img)
# # print(np.asarray(img_open))
# img_cropped = crop_innersqr(img_open)
# img_reshaped = img_cropped.resize(resolution)
# print(process_image)
arr = process_image(path_img, (64, 64))
print(arr)

arr2 = process_image(path_img, (64, 64), "CHW")
print(arr2)


# arr_reformat = reformat_image(arr)
# print(arr_reformat)

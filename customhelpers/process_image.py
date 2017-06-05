#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in modules
import os

# Import 3rd party packages
import pickle as pickle
import numpy as np
import PIL.Image as Image

# Define some constants
DIR_DATA = "C:\\dev\\lab_fda\\data\\kaggle_catdog\\"


def make_container(num_rows, num_cols, dtype=np.int8):
	container = np.zeros(shape=(num_rows, num_cols), dtype=np.int8)
	return container

def fwrite_pickle(filepath_pickle, container_data):
	with open(filepath_pickle, 'wb') as handle:
		pickle.dump(container_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#############################################################################
container_data = make_container(20000, 64*64*3 + 2)

seq_filepath = os.listdir(DIR_DATA + "train\\")
print(seq_filepath)
for i, filepath in enumerate(seq_filepath):
	path_image = DIR_DATA + "train\\" + filepath
	shape = (64,64)

	img_open = Image.open(path_image)
	img_resized = img_open.resize(shape)
	arr_img_resized = np.asarray(img_resized)
	arr_img_ser = arr_img_resized.reshape(1, shape[0]*shape[1]*3)[0]

	if 'cat' in filepath:
		vec_class = [1, 0]
	elif 'dog' in filepath:
		vec_class = [0, 1]
	container_data[i] = np.append(arr_img_ser, vec_class)

	print(i)


# for i in range(0,10000):
# 	path_image = DIR_DATA + "train\\cat." + str(i) + ".jpg"
# 	shape = (64,64)

# 	img_open = Image.open(path_image)
# 	img_resized = img_open.resize(shape)
# 	arr_img_resized = np.asarray(img_resized)
# 	arr_img_ser = arr_img_resized.reshape(1, shape[0]*shape[1]*3)[0]

# 	container_data[i] = np.append(arr_img_ser, [1, 0])

# 	print(i)

# for j in range(0,10000):
# 	path_image = DIR_DATA + "train\\dog." + str(j) + ".jpg"
# 	shape = (64,64)

# 	img_open = Image.open(path_image)
# 	img_resized = img_open.resize(shape)
# 	arr_img_resized = np.asarray(img_resized)
# 	arr_img_ser = arr_img_resized.reshape(1, shape[0]*shape[1]*3)[0]

# 	container_data[10000 + j] = np.append(arr_img_ser, [0, 1])

# 	print(j)

fwrite_pickle(DIR_DATA + "kaggle_catdog_train_64x64_TEST.pickle", container_data)

# for i in range(0,2500):
# 	path_image = DIR_DATA + "test\\cat." + str(10000 + i) + ".jpg"
# 	shape = (64,64)

# 	img_open = Image.open(path_image)
# 	img_resized = img_open.resize(shape)
# 	arr_img_resized = np.asarray(img_resized)
# 	arr_img_ser = arr_img_resized.reshape(1, shape[0]*shape[1]*3)[0]

# 	container_data[i] = np.append(arr_img_ser, [1, 0])

# 	print(i)

# for j in range(0,2500):
# 	path_image = DIR_DATA + "test\\dog." + str(10000 + j) + ".jpg"
# 	shape = (64,64)

# 	img_open = Image.open(path_image)
# 	img_resized = img_open.resize(shape)
# 	arr_img_resized = np.asarray(img_resized)
# 	arr_img_ser = arr_img_resized.reshape(1, shape[0]*shape[1]*3)[0]

# 	container_data[2500 + j] = np.append(arr_img_ser, [0, 1])

# 	print(j)

seq_filepath = os.listdir(DIR_DATA + "test\\")
print(seq_filepath)
for i, filepath in enumerate(seq_filepath):
	path_image = DIR_DATA + "test\\" + filepath
	shape = (64,64)

	img_open = Image.open(path_image)
	img_resized = img_open.resize(shape)
	arr_img_resized = np.asarray(img_resized)
	arr_img_ser = arr_img_resized.reshape(1, shape[0]*shape[1]*3)[0]
	
	if 'cat' in filepath:
		vec_class = [1, 0]
	elif 'dog' in filepath:
		vec_class = [0, 1]
	container_data[i] = np.append(arr_img_ser, vec_class)

	print(i)


fwrite_pickle(DIR_DATA + "kaggle_catdog_test_64x64_TEST.pickle",container_data)

#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os, sys

# Import 3rd party packages
import _pickle as cPickle
import pickle as pickle
import numpy as np
import PIL.Image as Image

SYSTEM = sys.platform
print(SYSTEM)

def get_direc_data_processed(system):
	if "linux" in system:
		direc = "/usr/local/dev-data/classifcation-cnn/kaggle_catdog/"
	elif "win" in system:
		direc = "C:\\dev\\lab_fda\\data\\kaggle_catdog\\"
	return direc

def open_image(path_img):
	ret = Image.open(path_img)
	return ret

def crop_innersqr(img_open):
	min_side = min(img_open.size)
	padding_h, padding_v = (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
	ret = img_open.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))
	return ret

def reshape_image(img_open, shape):
	img_open = img_open.resize(shape)
	return ret

def serialize_image(img_open):
	img_serialized = np.asarray(img_open)
	ret = img_serialized.reshape(1, img_open.size)[0]
	# ret = arr_img.reshape(1, 12288)[0]
	return ret

def process_image(path_img, resolution):
	img_open = open_image(path_img)
	img_cropped = crop_innersqr(img_open)
	img_reshaped = resize_image(img_cropped, resolution)
	ret = serialize_image(img_reshaped)		
	return ret

seq_pickle = np.zeros(shape=(20000, 64*64*3 + 1))

isIndex = True
for i in range(0,10):
	try: 
		path_img = direc + "train\\cat." + str(i) + ".jpg"
		img_processed =  process_image(path_img, resolution)
		seq_pickle[i] = np.append(img_processed, [3])
	except FileNotFoundError:
		isIndex = False
	print(i)

# for j in range(0,10):
# 	try: 
# 		kaggle_catdog.open_image(direc + "train\\dog." + str(j) + ".jpg")
# 		size_img = kaggle_catdog.image.size
# 		min_side = min(size_img)
# 		padding_h = (size_img[0] - min_side)/2
# 		padding_v = (size_img[1] - min_side)/2
# 		kaggle_catdog.image.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))

# 		kaggle_catdog.adjust_size((64,64))
# 		kaggle_catdog.serialize_image2()
# 		saechi_ser = kaggle_catdog.arr_img

# 		saechi_ser = np.append(saechi_ser, [5])
# 		seq_pickle[10000+j] = saechi_ser
# 	except FileNotFoundError:
# 		isIndex = False
# 	# j += 1
# 	print(j)

direc = get_direc_data_processed(system)
with open(direc + "kaggle_catdog_train_64x64_TEST.pickle", 'wb') as handle:
    pickle.dump(seq_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

# for i in range(0,2):
# 	try: 
# 		kaggle_catdog.open_image(direc + "test\\cat." + str(10000+i) + ".jpg")
# 		size_img = kaggle_catdog.image.size
# 		min_side = min(size_img)
# 		padding_h = (size_img[0] - min_side)/2
# 		padding_v = (size_img[1] - min_side)/2
# 		kaggle_catdog.image.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))

# 		kaggle_catdog.adjust_size((64,64))
# 		kaggle_catdog.serialize_image2()
# 		saechi_ser = kaggle_catdog.arr_img

# 		saechi_ser = np.append(saechi_ser, [3])
# 		seq_pickle[i] = saechi_ser
# 	except FileNotFoundError:
# 		isIndex = False
# 	# i += 1
# 	print(i)

# for j in range(0,2):
# 	try: 
# 		kaggle_catdog.open_image(direc + "test\\dog." + str(10000+j) + ".jpg")
# 		size_img = kaggle_catdog.image.size
# 		min_side = min(size_img)
# 		padding_h = (size_img[0] - min_side)/2
# 		padding_v = (size_img[1] - min_side)/2
# 		aa = kaggle_catdog.image.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))

# 		kaggle_catdog.adjust_size((64,64))
# 		kaggle_catdog.serialize_image2()
# 		saechi_ser = kaggle_catdog.arr_img

# 		saechi_ser = np.append(saechi_ser, [5])
# 		seq_pickle[2500+j] = saechi_ser
# 	except FileNotFoundError:
# 		isIndex = False
# 	# j += 1
# 	print(j)

# with open(direc + "kaggle_catdog_test_64x64_TEST.pickle", 'wb') as handle:
#     pickle.dump(seq_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
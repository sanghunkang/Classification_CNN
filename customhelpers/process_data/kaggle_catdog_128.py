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

from image_handler import process_image

SYSTEM = sys.platform
print(SYSTEM)

def get_direc_data_processed(system):
	if "linux" in system:
		direc = "/usr/local/dev-data/classifcation-cnn/kaggle_catdog"
	elif "win" in system:
		direc = "C:\\dev\\lab_fda\\data\\kaggle_catdog"
	return direc

def make_pickle(data_input, path_file):
	with open(path_file, 'wb') as handle:
		pickle.dump(data_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

direc = get_direc_data_processed(SYSTEM)

seq_pickle = np.zeros(shape=(20000, 64*64*3 + 1))

isIndex = True
for i in range(0,10000):
	# try: 
	path_img = direc + "\\train\\cat." + str(i) + ".jpg"
	img_processed =  process_image(path_img, (64,64))
	seq_pickle[i] = np.append(img_processed, [3])
	# except FileNotFoundError:
	# 	isIndex = False
	print(i)

for j in range(0,10000):
	# try: 
	path_img = direc + "\\train\\dog." + str(j) + ".jpg"
	img_processed =  process_image(path_img, (64,64))
	seq_pickle[10000+j] = np.append(img_processed, [5])
	# except FileNotFoundError:
	# 	isIndex = False
	print(j)

make_pickle(seq_pickle, direc + "\\kaggle_catdog_train_64x64_TEST.pickle")
# with open(direc + "\\kaggle_catdog_train_64x64_TEST.pickle", 'wb') as handle:
#     pickle.dump(seq_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
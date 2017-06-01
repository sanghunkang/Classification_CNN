#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os, sys

# Import 3rd party packages
import pickle as pickle
import numpy as np

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

for i in range(0,10000): 
	path_img = direc + "\\train\\cat." + str(i) + ".jpg"
	img_processed =  process_image(path_img, (64,64))
	seq_pickle[i] = np.append(img_processed, [3])
	print(i)

for j in range(0,10000):
	path_img = direc + "\\train\\dog." + str(j) + ".jpg"
	img_processed =  process_image(path_img, (64,64))
	seq_pickle[10000+j] = np.append(img_processed, [5])
	print(j)

make_pickle(seq_pickle, direc + "\\kaggle_catdog_train_64x64_TEST.pickle")

for i in range(0,2500):
	path_img = direc + "test\\cat." + str(10000+i) + ".jpg"
	img_processed =  process_image(path_img, (64,64))
	seq_pickle[i] = np.append(img_processed, [3])
	print(i)

for j in range(0,2500):
	path_img = direc + "test\\dog." + str(10000+j) + ".jpg"
	img_processed =  process_image(path_img, (64,64))
	seq_pickle[i] = np.append(img_processed, [5])
	seq_pickle[2500+j] = saechi_ser
	print(j)

make_pickle(seq_pickle, direc + "\\kaggle_catdog_test_64x64_TEST.pickle")
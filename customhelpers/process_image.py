#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in modules
import os

# Import 3rd party packages
import _pickle as cPickle
import pickle as pickle
import numpy as np
import PIL.Image as Image

from image_handler import Image_handler
direc = "C:\\dev\\lab_fda\\data\\kaggle_catdog\\"
kaggle_catdog = Image_handler()

seq_pickle = np.zeros(shape=(20000,64*64*3 + 1), dtype=np.int8)

#############################################################################
for i in range(0,10000):
	# open_image
	path_image = direc + "train\\cat." + str(i) + ".jpg"
	img_open = Image.open(path_image)
	arr_img = np.asarray(img_open)
	# kaggle_catdog.open_image(direc + "train\\cat." + str(i) + ".jpg")

	# adjust_size
	shape = (64,64)
	img_resized = img_open.resize(shape)
	arr_img_resized = np.asarray(img_resized)
	# kaggle_catdog.adjust_size((64,64))
	
	# serialise image
	arr_img_ser = arr_img_resized.reshape(1, 64*64*3)[0]
	# kaggle_catdog.serialize_image2()
	
	# print(arr_img_ser)
	# print(kaggle_catdog.arr_img)
	# print('+++++++++++++++++++++++++++++')

	seq_pickle[i] = np.append(arr_img_ser, [3])

	print(i)

for j in range(0,10000):
	try: 
		kaggle_catdog.open_image(direc + "train\\dog." + str(j) + ".jpg")
		size_img = kaggle_catdog.image.size
		min_side = min(size_img)
		padding_h = (size_img[0] - min_side)/2
		padding_v = (size_img[1] - min_side)/2
		aa = kaggle_catdog.image.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))

		kaggle_catdog.adjust_size((64,64))
		# kaggle_catdog.serialize_image(64*64,3)
		kaggle_catdog.serialize_image2()
		saechi_ser = kaggle_catdog.arr_img

		saechi_ser = np.append(saechi_ser, [5])
		seq_pickle[10000+j] = saechi_ser
	except FileNotFoundError:
		isIndex = False
	# j += 1
	print(j)

with open(direc + "kaggle_catdog_train_64x64_TEST.pickle", 'wb') as handle:
    pickle.dump(seq_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

# seq_pickle = np.zeros(shape=(5000,64*64*3 + 1))

for i in range(0,2500):
	try: 
		kaggle_catdog.open_image(direc + "test\\cat." + str(10000+i) + ".jpg")
		size_img = kaggle_catdog.image.size
		min_side = min(size_img)
		padding_h = (size_img[0] - min_side)/2
		padding_v = (size_img[1] - min_side)/2
		aa = kaggle_catdog.image.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))

		kaggle_catdog.adjust_size((64,64))
		# kaggle_catdog.serialize_image(64*64,3)
		kaggle_catdog.serialize_image2()
		saechi_ser = kaggle_catdog.arr_img
		# show_from_serialized_img(saechi_ser, (64, 64))

		saechi_ser = np.append(saechi_ser, [3])
		seq_pickle[i] = saechi_ser
	except FileNotFoundError:
		isIndex = False
	# i += 1
	print(i)

for j in range(0,2500):
	try: 
		kaggle_catdog.open_image(direc + "test\\dog." + str(10000+j) + ".jpg")
		size_img = kaggle_catdog.image.size
		min_side = min(size_img)
		padding_h = (size_img[0] - min_side)/2
		padding_v = (size_img[1] - min_side)/2
		aa = kaggle_catdog.image.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))

		kaggle_catdog.adjust_size((64,64))
		# kaggle_catdog.serialize_image(64*64,3)
		kaggle_catdog.serialize_image2()
		saechi_ser = kaggle_catdog.arr_img

		saechi_ser = np.append(saechi_ser, [5])
		seq_pickle[2500+j] = saechi_ser
	except FileNotFoundError:
		isIndex = False
	# j += 1
	print(j)

with open(direc + "kaggle_catdog_test_64x64_TEST.pickle", 'wb') as handle:
    pickle.dump(seq_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
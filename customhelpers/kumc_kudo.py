#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os

# Import 3rd party packages
import _pickle as cPickle
import pickle as pickle
import numpy as np
import PIL.Image as Image

import psycopg2
import psycopg2.extensions
import logging

import psycopg2



# Connect to an existing database
conn = psycopg2.connect("dbname=kumc_colon_proto user=postgres password=0000")

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a command: this creates a new table
# cur.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data varchar);")

# Pass data to fill a query placeholders and let Psycopg perform
# the correct conversion (no more SQL injections!)
x = np.array([[0, 4], [2, 3]])
xx = psycopg2.Binary(x)

direc = os.path.dirname(os.path.dirname(os.getcwd())) +"\\data\\Kaggle_catdog\\train\\"

cur.execute("INSERT INTO train_set (x, y) VALUES ({}, {})".format(xx, 0))




# direc = os.path.dirname(os.path.dirname(os.getcwd())) +"\\data\\KUMC_Kudo"

# print(direc)

i = 0
filenames = os.listdir(direc)
for filename in filenames:
	im = Image.open(direc + filename)
	im = im.resize((128, 128))
	arr = np.asarray(im)

	if filename.split("_")[0] == "t1":
		arr = np.append(arr, [0])
		seq_pickle[i] = arr
	elif filename.split("_")[0] == "t2":
		arr = np.append(arr, [1])
		seq_pickle[i] = arr
	print(type(arr))
	i += 1
	if filename.split("_")[0] in ['t1']:
	
	for i in range(360):
		# Calculate cropping range
		width, height = im.size   # Get dimensions
		
		new_width = width/2
		new_height = height/2
		
		left = (width - new_width)/2
		top = (height - new_height)/2
		right = (width + new_width)/2
		bottom = (height + new_height)/2

		# Rotate and crop the original image
		im_rot = im.rotate(i)
		im_rot = im_rot.crop((left, top, right, bottom))
		fn = direc + "\\augmented\\" + filename.replace(".jpg", "") + "_rot" + str(i) + ".jpg"
		im_rot.save(fn)
	

	# Make the changes to the database persistent
	conn.commit()
	print(fn)

# seq_pickle = np.zeros(shape=(12960,64*64*3 + 1))

# filenames = os.listdir(direc + "\\augmented")
# i = 0
# for filename in filenames:
# 	im = Image.open(direc + "\\augmented\\"+ filename)
# 	im = im.resize((64,64))
# 	arr = np.asarray(im)

# 	if filename.split("_")[0] == "t1":
# 		arr = np.append(arr, [0])
# 		seq_pickle[i] = arr
# 	elif filename.split("_")[0] == "t2":
# 		arr = np.append(arr, [1])
# 		seq_pickle[i] = arr
# 	print(arr)
# 	i += 1



# print(direc)
# with open(direc + "\\KUMC_kudo_train_64x64.pickle", 'wb') as handle:
#     pickle.dump(seq_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Query the database and obtain data as Python objects
# cur.execute("SELECT * FROM train_set")
# cur.fetchone()


# Close communication with the database
cur.close()
conn.close()
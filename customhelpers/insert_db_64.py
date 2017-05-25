#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os

# Import 3rd party packages
import numpy as np
import PIL.Image as Image
import psycopg2 as psycopg2

conn = psycopg2.connect("host=localhost dbname=kaggle_catdog user=postgres password=0000")
# conn = psycopg2.connect("host=localhost dbname=kumc_colon_proto user=postgres password=0000")
cur = conn.cursor()

i = 0
for name_set in ['train','test']:
	direc = os.path.dirname(os.path.dirname(os.getcwd())) +"\\data\\Kaggle_catdog\\" + name_set + "\\"
	# direc = os.path.dirname(os.path.dirname(os.getcwd())) +"\\data\\KUMC_Kudo"

	filenames = os.listdir(direc)
	for filename in filenames:
		im = Image.open(direc + filename)
		
		# Calculate cropping range
		width, height = im.size   # Get dimensions
		new_edge = min(width, height)
		# new_width = width/2
		# new_height = height/2

		left = (width - new_edge)/2
		top = (height - new_edge)/2
		right = (width + new_edge)/2
		bottom = (height + new_edge)/2

		im_sqr = im.crop((left, top, right, bottom))
		# im_sqr = im.crop((left, top, left + 224, top + 224))

		im_sqr = im_sqr.resize((64, 64))
		arr = np.asarray(im_sqr)
		arr = arr.astype(np.int64)
		# arr = arr.reshape(1, arr.size)
	
		if ('cat' in filename):
			sql = cur.mogrify("INSERT INTO " + name_set + "_set (x, y) VALUES ({}, {})".format(psycopg2.Binary(arr), 0))
		elif ('dog' in filename):
			sql = cur.mogrify("INSERT INTO " + name_set + "_set (x, y) VALUES ({}, {})".format(psycopg2.Binary(arr), 1))
		
		cur.execute(sql)
		conn.commit()

		i += 1
		print(i)
		# cur.execute(sql)

cur.close()
conn.close()
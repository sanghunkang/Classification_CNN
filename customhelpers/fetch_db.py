#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os
import _pickle as cPickle
import pickle as pickle

# Import 3rd party packages
import numpy as np
import psycopg2 as psycopg2

# Import custom packages

direc = os.path.dirname(os.path.dirname(os.getcwd())) +"\\data\\Kaggle_catdog"
print(direc)

conn = psycopg2.connect("host=localhost dbname=kaggle_catdog user=postgres password=0000")
# conn = psycopg2.connect("dbname=kumc_colon_proto user=postgres password=0000")
cur = conn.cursor()



seq_pickle = np.zeros(shape=(5000,128*128*3 + 2))
j = 0
for i in list(range(1,2501)) + list(range(10001, 12501)):
	cur.execute("SELECT x_y FROM train_set WHERE id={}".format(i))
	rec_fetch = cur.fetchone()
	batch = np.frombuffer(rec_fetch[0], np.int64)
	seq_pickle[j] = batch
	print(str(j) + " "+ str(batch))
	j += 1

with open(direc + "\\Kaggle_catdot_train_128x128_1.pickle", 'wb') as handle:
    pickle.dump(seq_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)



seq_pickle = np.zeros(shape=(5000,128*128*3 + 2))
j = 0
for i in list(range(2501, 5001)) + list(range(12501, 15001)):
	cur.execute("SELECT x_y FROM train_set WHERE id={}".format(i))
	rec_fetch = cur.fetchone()
	batch = np.frombuffer(rec_fetch[0], np.int64)
	seq_pickle[j] = batch
	print(str(j) + " "+ str(batch))
	j += 1

with open(direc + "\\Kaggle_catdog_train_128x128_2.pickle", 'wb') as handle:
    pickle.dump(seq_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)


seq_pickle = np.zeros(shape=(5000,128*128*3 + 2))
j = 0
for i in list(range(5001, 7501)) + list(range(15001, 17501)):
	cur.execute("SELECT x_y FROM train_set WHERE id={}".format(i))
	rec_fetch = cur.fetchone()
	batch = np.frombuffer(rec_fetch[0], np.int64)
	seq_pickle[j] = batch
	print(str(j) + " "+ str(batch))
	j += 1

with open(direc + "\\Kaggle_catdog_train_128x128_3.pickle", 'wb') as handle:
    pickle.dump(seq_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)


seq_pickle = np.zeros(shape=(5000,128*128*3 + 2))
j = 0
for i in list(range(7501, 10001)) + list(range(17501, 20001)):
	cur.execute("SELECT x_y FROM train_set WHERE id={}".format(i))
	rec_fetch = cur.fetchone()
	batch = np.frombuffer(rec_fetch[0], np.int64)
	seq_pickle[j] = batch
	print(str(j) + " "+ str(batch))
	j += 1

with open(direc + "\\Kaggle_catdog_train_128x128_4.pickle", 'wb') as handle:
    pickle.dump(seq_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

cur.close()
conn.close()
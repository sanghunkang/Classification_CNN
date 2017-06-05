#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in modules
import _pickle as cPickle

# Import external packages
import numpy as np

def import_data(filename):
	with open(filename, "rb") as fo:
		data = cPickle.load(fo, encoding="bytes")
		np.random.shuffle(data)
		return data
#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import 3rd party packages
import matplotlib.pyplot as plt
import numpy as np

# Functions for messy computations
def encode_onehot(seq, n_classes, zero_columns=True):
    # Determine shape of the output one-hot matrix
    n_rec = len(seq)
    n_classes = n_classes

    # Create the output one-hot matrix 
    mat = np.zeros((n_rec, n_classes))
    mat[np.arange(n_rec), seq] = 1

    # Omit all-zero columns if specified
    if zero_columns == False:
        ret = mat[:, np.apply_along_axis(np.count_nonzero, 0, mat) > 0]
    
    return ret

def show_from_serialized_img(img_ser, channel, shape):
    # size_img = reduce(lambda x1,x2: x1*x2, *shape)
    

    if channel == 1:
        size_img = shape[0]*shape[1]
        img_rgb_tp = img_ser.reshape(size_img)

        img_rgb = img_rgb_tp.transpose()
        img_rgb = img_rgb.reshape(shape[0], shape[1])
        img_rgb = img_rgb/256
        img_rgb = img_rgb.astype(np.float32, copy=False)

        plt.imshow(img_rgb, cmap='gray')
        plt.show()
    else:
        size_img = shape[0]*shape[1]
        img_rgb_tp = img_ser.reshape(channel,size_img)

        img_rgb = img_rgb_tp.transpose()
        img_rgb = img_rgb.reshape(shape[0], shape[1], channel)
        img_rgb = img_rgb/256
        img_rgb = img_rgb.astype(np.float32, copy=False)

        plt.imshow(img_rgb)
        plt.show()


#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
import time
import scipy.io
import glob
import pickle
from keras.utils.np_utils import to_categorical
from keras import backend as K
import tensorflow as tf
import keras
# import cv2
import cnn_240599_loss_pid_sigmoid as cnn
size=22050

# set fraction of GPU the program can use

# set the value of data format convention
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# handles for txt files

# read into arousal data (arousal data is only helper data, help to determine the sleep length)

# load models
all_models=glob.glob('weights_*.h5')
for the_model in all_models:
    model = cnn.cnn1d(size,1)
    model.load_weights(the_model)
#    path1=the_path
 #   the_file=the_file.replace('train','test')
    all_test_files=open('test_gs','r')
    PRED=open(('prediction.dat.'+the_model),'w')
    for test_line in all_test_files:
        sample = test_line
        table=sample.split('\t')
        the_id=table[0]
        image = np.load('/home/shdong/cough/data/code/processedfile_solicited/' + the_id + '.npy')#,delimiter=',',skiprows=1)[:,1:4]
        #the_mean=np.mean(image,axis=0)
        #the_std=np.std(image,axis=0)
        #image=(image-the_mean)/the_std
        image_pad=np.zeros((size,1))
        image_pad[0:image.shape[0],0]=image
        image_batch=[]
        image_batch.append(image_pad)
        image_batch=np.asarray(image_batch)

        addition_batch=[]
        addition = float(table[2])
        addition_batch.append(addition)
        addition_batch=np.asarray(addition_batch)
        output = model.predict([image_batch, addition_batch])
            # below are not modified
        PRED.write('%.4f\n' % output)
    PRED.close()

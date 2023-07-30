import librosa
import numpy as np
import glob
import os
all_file=glob.glob('input_norm/*')
os.system('mkdir mfcc_norm')

for the_file in all_file:
    t=the_file.split('/')
    y, sr = librosa.load(the_file)
    mfcc=np.zeros((20,22))
    mfcc_value=librosa.feature.mfcc(y=y, sr=sr)
    mfcc[0:mfcc_value.shape[0],0:mfcc_value.shape[1]]=mfcc_value
    mfcc=mfcc.flatten()
    #zero_crossing=np.zeros((1,22))
    #zero_crossing_value=librosa.feature.zero_crossing_rate(y)
    #zero_crossing[zero_crossing_value.shape[0],zero_crossing_value.shape[1]]=zero_crossing_value
    np.save('mfcc_norm/'+t[-1],mfcc)


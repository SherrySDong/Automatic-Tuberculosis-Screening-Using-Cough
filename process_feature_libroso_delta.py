import librosa
import numpy as np
import glob
import os

all_file=glob.glob('/input/raw_test_data/*')
os.system('mkdir delta')

for the_file in all_file:
    t=the_file.split('/')
    y, sr = librosa.load(the_file)
#    print(librosa.feature.mfcc(y=y, sr=sr).shape)
    mfcc=np.zeros((20,22))
    mfcc_tmp=librosa.feature.mfcc(y=y, sr=sr)
    mfcc_tmp=librosa.feature.delta(mfcc_tmp)

    mfcc[0:mfcc_tmp.shape[0],0:mfcc_tmp.shape[1]]=mfcc_tmp
    mfcc=mfcc.flatten()
    #zero_crossing=np.zero((1,22))
    #zero_crossing_tmp=librosa.feature.zero_crossing_rate(y)
    #zero_crossing[zero_crossing_tmp.shape[0],zero_crossing_tmp.shape[1]]=zero_crossing_tmp
    np.save('delta/'+t[-1],mfcc)


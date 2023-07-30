import librosa
import numpy as np
import glob
import os
all_file=glob.glob('input_norm/*')
os.system("mkdir magnitude_norm")

for the_file in all_file:
    t=the_file.split('/')
    y, sr = librosa.load(the_file)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    mfcc=np.zeros((1025,22))
    mfcc_tmp, mag_abc=librosa.piptrack(y=y, sr=sr)
    mfcc[0:mag_abc.shape[0],0:mag_abc.shape[1]]=mag_abc
    mfcc=mfcc.flatten()
    np.save('magnitude_norm/'+t[-1],mfcc)


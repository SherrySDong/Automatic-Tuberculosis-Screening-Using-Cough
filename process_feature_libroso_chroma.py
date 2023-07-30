import librosa
import numpy as np
import glob
import os
all_file=glob.glob('/input/raw_test_data/*')
#all_file=glob.glob('../data/data/solicited/*')
os.system("mkdir chroma")

for the_file in all_file:
    t=the_file.split('/')
    y, sr = librosa.load(the_file)
#    print(librosa.feature.chroma_stft(y=y, sr=sr).shape)
    chroma=np.zeros((12,22))
    chroma_value=librosa.feature.chroma_stft(y=y, sr=sr)
    chroma[0:chroma_value.shape[0],0:chroma_value.shape[1]]=chroma_value
    chroma=chroma.flatten()
    #zero_crossing=np.zero((1,22))
    #zero_crossing_value=librosa.feature.zero_crossing_rate(y)
    #zero_crossing[zero_crossing_value.shape[0],zero_crossing_value.shape[1]]=zero_crossing_value
    np.save('chroma/'+t[-1],chroma)


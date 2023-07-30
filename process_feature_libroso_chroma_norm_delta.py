import librosa
import numpy as np
import glob
import os
os.system('mkdir chroma_norm_delta')
all_file=glob.glob('input_norm/*')


for the_file in all_file:
    t=the_file.split('/')
    y, sr = librosa.load(the_file)
    print(librosa.feature.chroma_stft(y=y, sr=sr).shape)
    delta=np.zeros((12,22))
    delta_value=librosa.feature.chroma_stft(y=y, sr=sr)
    delta_value=librosa.feature.delta(delta_value)
    delta[0:delta_value.shape[0],0:delta_value.shape[1]]=delta_value
    delta=delta.flatten()
    #zero_crossing=np.zero((1,22))
    #zero_crossing_value=librosa.feature.zero_crossing_rate(y)
    #zero_crossing[zero_crossing_value.shape[0],zero_crossing_value.shape[1]]=zero_crossing_value
    np.save('chroma_norm_delta/'+t[-1],delta)


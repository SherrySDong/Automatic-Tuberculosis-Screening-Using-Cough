import glob
all_file=glob.glob("/input/raw_test_data/*.wav")
from pydub import AudioSegment, effects  
import os
os.system('mkdir input_norm')
for the_file in all_file:
    findname=the_file.split('/')
    rawsound = AudioSegment.from_file(the_file, "wav")  
    normalizedsound = effects.normalize(rawsound)  
    normalizedsound.export("input_norm/"+findname[-1], format="wav")

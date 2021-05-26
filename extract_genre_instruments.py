from scipy.io import wavfile
import numpy as np
import os, glob
import shutil
from pydub import AudioSegment

class Extractor():
    def __init__(self,folder="./sleepsongs"):
        self.path=folder
        
    def extract(self):
        for filename in glob.glob(os.path.join(self.path, '*.wav')):
            print("\nPerforming source separation on "+filename)
            os.system(f"python3 -m demucs.separate '{filename}' -d cuda --dl")
        for filename in glob.glob(os.path.join(self.path, '*.mp3')):
            print("\nPerforming source separation on "+filename)
            os.system(f"python3 -m demucs.separate '{filename}' -d cuda --dl")
            
    def sort_instruments(self,path="./separated/demucs",instrument="other"):
        c=0
        if not os.path.exists("separated/demucs/"+instrument):
            os.system("mkdir ./separated/demucs/"+instrument)
        else:
            print("Colletion for this instrument already exists")
            return
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if file[:5]==instrument:
                    old_name=os.path.join(subdir, file)
                    sound = AudioSegment.from_wav(old_name)
                    sound = sound.set_channels(1)
                    new_name="./separated/demucs/"+instrument+"/"+str(c)+".wav"
                    #shutil.copy(old_name, new_name)
                    sound.export(new_name, format="wav")
                    c+=1

Extractor().extract()       
Extractor().sort_instruments()
Extractor().sort_instruments(instrument="vocal")
Extractor().sort_instruments(instrument="drums")
Extractor().sort_instruments(instrument="bass")
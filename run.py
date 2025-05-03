import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import sys
import pdb
from IPython import embed
import os
import time

for i in range(13):

    cmd = f"python feature_extraction1.py wav2vec2B {str(i)} 768 binary"
    os.system(cmd)


for i in range(25):
    cmd = f"python feature_extraction1.py wav2vec2L {str(i)} 1024 binary"
    os.system(cmd)


for i in range(25):
    cmd = f"python feature_extraction1.py hubert {str(i)} 1024 binary"
    os.system(cmd)
    

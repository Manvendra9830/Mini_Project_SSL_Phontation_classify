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
    cmd = f"python classification.py wav2vec2B {str(i)} binary"
    os.system(cmd)


for i in range(25):
    cmd = f"python classification.py wav2vec2L {str(i)} binary"
    os.system(cmd)

for i in range(25):
    cmd = f"python classification.py hubert {str(i)} binary"
    os.system(cmd)
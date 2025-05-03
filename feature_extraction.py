import sys
from tqdm import tqdm
from IPython import embed
import matplotlib.pyplot as plt
import sys
import librosa
import json
import os
import numpy as np
import pickle
import glob
from tqdm import tqdm
from IPython import embed
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC, WhisperForConditionalGeneration, WhisperProcessor
import torch
import time


def extract_pretrained(model_checkpoint, device, cache_dir_model):
    
    # load the pretrained model
    if model_checkpoint == 'facebook/wav2vec2-base-960h' or model_checkpoint == 'facebook/wav2vec2-large-960h-lv60-self':
        model = Wav2Vec2ForCTC.from_pretrained(model_checkpoint, cache_dir=cache_dir_model).to(device)
        processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)

    elif model_checkpoint == 'facebook/hubert-large-ls960-ft':
        model = HubertForCTC.from_pretrained(model_checkpoint, cache_dir=cache_dir_model).to(device)
        processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)
        
    
#  create dict for saving features in .pickle file
    feat_dict = {}

    x_h = []
    y_h = []
    for f_name in tqdm(os.listdir(ROOT_DIR_H)):

        # get the filenames with path
        wav_file_path = os.path.join(ROOT_DIR_H, f_name)

        # load the data
        wav = librosa.core.load(wav_file_path, sr=16000)[0]

        # apply the processor on wav files
        input_value = processor(wav, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        
        with torch.no_grad():
            outputs = model(input_value.to(device), output_hidden_states=True)
        feature = outputs.hidden_states[LAYER].cpu().numpy()[0, :, :].T
        mean_feat = np.mean(feature, axis=1)

        # Check for signals that are entirely zeros or contain NaNs
        if np.all(mean_feat == 0) or np.isnan(mean_feat).any():
            print('##################################################')
            print(f_name)
            print('##################################################')

        assert mean_feat.ndim == 1, "Features must be 1D array"
        assert mean_feat.shape[0] == FEAT_DIM, "MFCC features must have 768 elements"

        x_h.append(mean_feat)
        y_h.append(0)

    # stack the list of features
    x_h = np.stack(x_h) 
    y_h = np.hstack(y_h) 

    x_b = []
    y_b = []
    for f_name in tqdm(os.listdir(ROOT_DIR_B)):

        # get the filenames with path
        wav_file_path = os.path.join(ROOT_DIR_B, f_name)

        # load the data
        wav = librosa.core.load(wav_file_path, sr=16000)[0]

        # apply the processor on wav files
        input_value = processor(wav, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        
        with torch.no_grad():
            outputs = model(input_value.to(device), output_hidden_states=True)
        feature = outputs.hidden_states[LAYER].cpu().numpy()[0, :, :].T
        mean_feat = np.mean(feature, axis=1)

        # Check for signals that are entirely zeros or contain NaNs
        if np.all(mean_feat == 0) or np.isnan(mean_feat).any():
            print('##################################################')
            print(f_name)
            print('##################################################')

        assert mean_feat.ndim == 1, "Features must be 1D array"
        assert mean_feat.shape[0] == FEAT_DIM, "MFCC features must have 768 elements"

        x_b.append(mean_feat)
        y_b.append(1)

    # stack the list of features
    x_b = np.stack(x_b) 
    y_b = np.hstack(y_b)

    x_p = []
    y_p = []
    for f_name in tqdm(os.listdir(ROOT_DIR_P)):

        # get the filenames with path
        wav_file_path = os.path.join(ROOT_DIR_P, f_name)

        # load the data
        wav = librosa.core.load(wav_file_path, sr=16000)[0]

        # apply the processor on wav files
        input_value = processor(wav, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        
        with torch.no_grad():
            outputs = model(input_value.to(device), output_hidden_states=True)
        feature = outputs.hidden_states[LAYER].cpu().numpy()[0, :, :].T
        mean_feat = np.mean(feature, axis=1)

        # Check for signals that are entirely zeros or contain NaNs
        if np.all(mean_feat == 0) or np.isnan(mean_feat).any():
            print('##################################################')
            print(f_name)
            print('##################################################')

        assert mean_feat.ndim == 1, "Features must be 1D array"
        assert mean_feat.shape[0] == FEAT_DIM, "MFCC features must have 768 elements"

        x_p.append(mean_feat)
        y_p.append(2)


    # stack the list of features
    x_p = np.stack(x_p) 
    y_p = np.hstack(y_p) 

    X = np.concatenate((x_h,x_b,x_p))
    Y = np.concatenate((y_h,y_b, y_p))

    feat_dict['x'] = X
    feat_dict['y'] = Y


    # Save the features to file
    with open(OUTPUT_PKL_PATH, "wb") as f:
        pickle.dump(feat_dict, f)


def main():

    extract_pretrained(model_checkpoint, device, cache_dir_model)
        

if __name__ == '__main__':
    model_name = sys.argv[1]   # wav2vec2B, wav2vec2L, hubert
    layer_num = sys.argv[2]    # 0,...,12, 0,...,24
    feat_dim = sys.argv[3]     # 768, 1024
    task_type = sys.argv[4]    # binary, multi, multiVD
 
    LAYER = int(layer_num)
    FEAT_DIM = int(feat_dim)

    # Paths Configuration
  
    ROOT_DIR_H = {'binary':r"data\normal"}[task_type]

    ROOT_DIR_B = {'binary':r"data\breathy"}[task_type]

    ROOT_DIR_P = {'binary':"data\pressed"}[task_type]

    
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #get model_checkpoint 
    model_checkpoint = {'wav2vec2B':'facebook/wav2vec2-base-960h',
                        'wav2vec2L':'facebook/wav2vec2-large-960h-lv60-self',
                        'hubert':'facebook/hubert-large-ls960-ft'}[model_name]

    #get model_checkpoint cache dir 
    cache_dir_model = "pretrained_model"


    # make folder for saving features
    feature_folder_path = os.path.join("features", task_type)


    # create features folder
    if not os.path.exists(feature_folder_path):
        os.makedirs(feature_folder_path)

    # output pkl file
    OUTPUT_PKL_PATH = os.path.join(feature_folder_path, f"{model_name}_{LAYER}.pkl")



    main()





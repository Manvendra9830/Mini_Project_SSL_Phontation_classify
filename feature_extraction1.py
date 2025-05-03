import sys
from tqdm import tqdm
from IPython import embed
import matplotlib.pyplot as plt
import librosa
import json
import os
import numpy as np
import pickle
import glob
import torch
import time
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, HubertForSequenceClassification

def extract_features(finetuned_model_dir, device):
    # Load the fine-tuned model and processor
    if "wav2vec2B" in finetuned_model_dir or "wav2vec2L" in finetuned_model_dir:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(finetuned_model_dir).to(device)
        processor = Wav2Vec2Processor.from_pretrained(finetuned_model_dir)
    elif "hubert" in finetuned_model_dir:
        model = HubertForSequenceClassification.from_pretrained(finetuned_model_dir).to(device)
        processor = Wav2Vec2Processor.from_pretrained(finetuned_model_dir)
    else:
        raise ValueError("Unsupported fine-tuned model directory")

    # Create dict for saving features in .pickle file
    feat_dict = {}

    x_h = []
    y_h = []
    for f_name in tqdm(os.listdir(ROOT_DIR_H)):
        wav_file_path = os.path.join(ROOT_DIR_H, f_name)
        wav = librosa.core.load(wav_file_path, sr=16000)[0]
        input_value = processor(wav, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        
        with torch.no_grad():
            outputs = model(input_value.to(device), output_hidden_states=True)
        feature = outputs.hidden_states[LAYER].cpu().numpy()[0, :, :].T
        mean_feat = np.mean(feature, axis=1)

        if np.all(mean_feat == 0) or np.isnan(mean_feat).any():
            print('##################################################')
            print(f_name)
            print('##################################################')

        assert mean_feat.ndim == 1, "Features must be 1D array"
        assert mean_feat.shape[0] == FEAT_DIM, f"Features must have {FEAT_DIM} elements"

        x_h.append(mean_feat)
        y_h.append(0)

    x_h = np.stack(x_h)
    y_h = np.hstack(y_h)

    x_b = []
    y_b = []
    for f_name in tqdm(os.listdir(ROOT_DIR_B)):
        wav_file_path = os.path.join(ROOT_DIR_B, f_name)
        wav = librosa.core.load(wav_file_path, sr=16000)[0]
        input_value = processor(wav, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        
        with torch.no_grad():
            outputs = model(input_value.to(device), output_hidden_states=True)
        feature = outputs.hidden_states[LAYER].cpu().numpy()[0, :, :].T
        mean_feat = np.mean(feature, axis=1)

        if np.all(mean_feat == 0) or np.isnan(mean_feat).any():
            print('##################################################')
            print(f_name)
            print('##################################################')

        assert mean_feat.ndim == 1, "Features must be 1D array"
        assert mean_feat.shape[0] == FEAT_DIM, f"Features must have {FEAT_DIM} elements"

        x_b.append(mean_feat)
        y_b.append(1)

    x_b = np.stack(x_b)
    y_b = np.hstack(y_b)

    x_p = []
    y_p = []
    for f_name in tqdm(os.listdir(ROOT_DIR_P)):
        wav_file_path = os.path.join(ROOT_DIR_P, f_name)
        wav = librosa.core.load(wav_file_path, sr=16000)[0]
        input_value = processor(wav, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        
        with torch.no_grad():
            outputs = model(input_value.to(device), output_hidden_states=True)
        feature = outputs.hidden_states[LAYER].cpu().numpy()[0, :, :].T
        mean_feat = np.mean(feature, axis=1)

        if np.all(mean_feat == 0) or np.isnan(mean_feat).any():
            print('##################################################')
            print(f_name)
            print('##################################################')

        assert mean_feat.ndim == 1, "Features must be 1D array"
        assert mean_feat.shape[0] == FEAT_DIM, f"Features must have {FEAT_DIM} elements"

        x_p.append(mean_feat)
        y_p.append(2)

    x_p = np.stack(x_p)
    y_p = np.hstack(y_p)

    X = np.concatenate((x_h, x_b, x_p))
    Y = np.concatenate((y_h, y_b, y_p))

    feat_dict['x'] = X
    feat_dict['y'] = Y

    # Save the features to file
    with open(OUTPUT_PKL_PATH, "wb") as f:
        pickle.dump(feat_dict, f)

def main():
    finetuned_model_dir = f"finetuned_models/{model_name}_{task_type}"
    extract_features(finetuned_model_dir, device)

if __name__ == '__main__':
    model_name = sys.argv[1]   # wav2vec2B, wav2vec2L, hubert
    layer_num = sys.argv[2]    # 0,...,12, 0,...,24
    feat_dim = sys.argv[3]     # 768, 1024
    task_type = sys.argv[4]    # binary, multi, multiVD

    LAYER = int(layer_num)
    FEAT_DIM = int(feat_dim)

    # Paths Configuration
    ROOT_DIR_H = {'binary': r"data\normal"}[task_type]
    ROOT_DIR_B = {'binary': r"data\breathy"}[task_type]
    ROOT_DIR_P = {'binary': r"data\pressed"}[task_type]

    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Make folder for saving features
    feature_folder_path = os.path.join("features_finetuned", task_type)

    # Create features folder
    if not os.path.exists(feature_folder_path):
        os.makedirs(feature_folder_path)

    # Output pkl file
    OUTPUT_PKL_PATH = os.path.join(feature_folder_path, f"{model_name}_{LAYER}.pkl")

    main()
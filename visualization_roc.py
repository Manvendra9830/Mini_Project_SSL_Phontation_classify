import matplotlib.pyplot as plt
import pickle
import pdb
import sys
from IPython import embed
debug = pdb.Pdb(stdout = sys.__stdout__).set_trace
import json
import pickle
import os
import numpy as np


def fpr_tpr_return(path):

    with open(path, "rb") as f:
        res = pickle.load(f)
    
    return res['fpr'], res['tpr']

def main():
    path_res = ["results/multi/wav2vec2L_1.pkl", "results/multi/hubert_6.pkl"]
    # path_res = ["results/multiVD/wav2vec2L_2.pkl", "results/multiVD/hubert_6.pkl"]
    # path_res = ["results/HUPA/wav2vec2L_5.pkl", "results/HUPA/hubert_1.pkl", "results/binary/wav2vec2L_5.pkl", "results/binary/hubert_4.pkl"]

    fpr0, tpr0 = fpr_tpr_return(path_res[0])
    fpr1, tpr1 = fpr_tpr_return(path_res[1])
    # fpr2, tpr2 = fpr_tpr_return(path_res[2])
    # fpr3, tpr3 = fpr_tpr_return(path_res[3])

    # plot ROC curve
    plt.subplots(1, figsize=(10,10))
    plt.title('ROC Curve', fontsize=16)
    plt.plot(fpr0, tpr0)
    plt.plot(fpr1, tpr1)
    # plt.plot(fpr2, tpr2)
    # plt.plot(fpr3, tpr3)
    plt.plot([0, 1], ls="--")
    # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.legend(["Wav2vec2-Large-2 (AUC = 0.85)", "Hubert-6 (AUC = 0.81)"], fontsize=14, loc="lower right")
    # plt.legend(["HUPA-Wav2vec2-Large-5 (AUC = 0.92)", "HUPA-Hubert-1 (AUC = 0.90)", "SVD-Wav2vec2-Large-5 (AUC = 0.86)", "SVD-Hubert-4 (AUC = 0.84)"], fontsize=14, loc="lower right")
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)

    plt.xticks(fontsize=14)  # Adjust fontsize for x-axis ticks
    plt.yticks(fontsize=14)  
    plt.show()



if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import pickle
import pdb
import sys
import os
import json
from IPython import embed
debug = pdb.Pdb(stdout = sys.__stdout__).set_trace

YLIM = (40,100)


def read_accuracy_std_from_path(path):
    """
    Reads and returns the std accuracy from 
    given result file path
    """
    with open(path, "rb") as f:
        return pickle.load(f)['Std_Acc']

def read_accuracy_mean_from_path(path):
    """
    Reads and returns the mean accuracy from 
    given result file path
    """
    with open(path, "r") as f:
        return json.load(f)['acc']

def main():
    """
    Creates the visUAlizations for
    the given result paths
    """
    task_type = sys.argv[1]    # binary, multi, multiVD
 
    PATH_B = [os.path.join("results1", task_type, f"wav2vec2B_{str(i)}.json") for i in range(13)]
    PATH_L = [os.path.join("results1", task_type, f"wav2vec2L_{str(i)}.json") for i in range(25)]
    PATH_H = [os.path.join("results1", task_type, f"hubert_{str(i)}.json") for i in range(25)]




    # VisUAlize
    plt.rc('axes', labelsize=14)
    plt.rc('font', size=12)
    plt.figure(figsize=(12,7))
    # legend_element_names = ["MFCCs(SVM)", "MFCCs(CNN)","OpenSMILE(SVM)","OpenSMILE(CNN)", "eGeMAPs(SVM)", "eGeMAPs(CNN)","wav2vec2-BASE(SVM)", "wav2vec2-BASE(CNN)","wav2vec2-LARGE(SVM)", 
    #                         "wav2vec2-LARGE(CNN)","HuBERT(SVM)", "HuBERT(CNN)"]
    
    legend_element_names = ["Wav2vec2-BASE", "Wav2vec2-LARGE", 
                            "HuBERT"]
    bar_labels = [str(i) for i in range(1, len(PATH_H)+1)]

    # bar_labels = [str(i) for i in range(1, len(RESULT_PATHS_HUBERT_SVM)+1)]
    # bar_heights = [read_accuracy_mean_from_path(SPEC_PATH)] + [read_accuracy_mean_from_path(MEL_PATH)] + [read_accuracy_mean_from_path(MFCC_PATH)]  + [read_accuracy_mean_from_path(path) for path in RESULT_PATHS]
    # yerrs = [read_accuracy_std_from_path(SPEC_PATH)] + [read_accuracy_std_from_path(MEL_PATH)] + [read_accuracy_std_from_path(MFCC_PATH)]  + [read_accuracy_std_from_path(path) for path in RESULT_PATHS]

    acc_b = [read_accuracy_mean_from_path(path) for path in PATH_B]

    acc_l = [read_accuracy_mean_from_path(path) for path in PATH_L]

    acc_h = [read_accuracy_mean_from_path(path) for path in PATH_H]



    ticks0 =[i for i in range(len(acc_b))]
    ticks1 =[i for i in range(len(acc_l))]
    ticks2 =[i for i in range(len(acc_h))]


    # plt.plot(ticks0, bar_heights_base_svm, linestyle="-.", marker='d', lw=2)
    plt.plot(ticks0, acc_b, linestyle="-.", marker='o', lw=2)

    # plt.plot(ticks1, bar_heights_large_svm, linestyle="-.", marker='d', lw=2)
    plt.plot(ticks2, acc_l, linestyle="-.", marker='o', lw=2)

    # plt.plot(ticks1, bar_heights_hubert_svm, linestyle="-.", marker='d', lw=2)
    plt.plot(ticks2, acc_h, linestyle="-.", marker='o', lw=2)
    plt.legend(legend_element_names, fontsize=13)

    plt.ylim(*YLIM)
    plt.title('Multi')
    plt.ylabel("Accuracy [%]")
    plt.xlabel("Feature")
    plt.xticks(ticks2, labels=bar_labels)
    plt.show()



if __name__ == "__main__":
    main()
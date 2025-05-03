import json
import os
import torch
from torch.utils.data import DataLoader
from IPython import embed
from torch.nn import functional
import matplotlib.pyplot as plt
import sys
import pickle
from IPython import embed
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
from tqdm import tqdm


def train(feat_data, skf, clf):

    X = feat_data['x']
    y = feat_data['y']

    # create lists for saving the evaluation metric 

    y_tests = []
    y_probs = []

    # train and test the classifier
    for train_indx, test_indx in tqdm(skf.split(X, y)):

        # get the x_train and y_train
        x_train, y_train = X[train_indx], y[train_indx]

        # get the x_test and y_test
        x_test, y_test = X[test_indx], y[test_indx]
        
        # standardize the data
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # Fit the data
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_prob = clf.predict_proba(x_test)

        y_probs.append(y_prob)
        y_tests.append(y_test)
        
    y_probs_app = np.vstack(y_probs)
    y_tests = np.hstack(y_tests)

    fpr, tpr, thresholds = metrics.roc_curve(y_tests, y_probs_app[:,1], pos_label=1.)
    

    results = {}
    results['fpr'] = fpr
    results['tpr'] = tpr


    with open(OUTPUT_RESULT_PATH, 'wb') as f_name:
        pickle.dump(results, f_name)



def main():

    # skf = StratifiedShuffleSplit(n_splits=10, random_state=42)
    skf = StratifiedKFold(n_splits=10, shuffle=False)


    # create the target classifier with defult parameters
    clf = svm.SVC(kernel='rbf', probability=True)
    # clf = RandomForestClassifier()
    # clf = AdaBoostClassifier()
    

    # load the features
    with open(FEAT_PKL_FILE, 'rb') as f:
        feat_data = pickle.load(f)

    train(feat_data, skf, clf)



if __name__ == '__main__':

    model_name = sys.argv[1]   # wav2vec2B, wav2vec2L, hubert
    layer_num = sys.argv[2]    # 0,...,12, 0,...,24
    task_type = sys.argv[3]    # binary, multi, multiVD, HUPA
 

    # make folder for saving features
    feature_folder_path = os.path.join("features", task_type)
    FEAT_PKL_FILE = os.path.join(feature_folder_path, f"{model_name}_{layer_num}.pkl")


    # make folder for saving features
    results_folder_path = os.path.join("results", task_type)


    # output pkl file
    OUTPUT_RESULT_PATH = os.path.join(results_folder_path, f"{model_name}_{layer_num}.pkl")

    # create features folder
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)


    main()

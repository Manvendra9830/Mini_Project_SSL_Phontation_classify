
import json
import os
import pickle
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def train(feat_data, skf, clf, params):
    X = feat_data['x']
    y = feat_data['y']

    # Create lists for saving the evaluation metrics
    acc = []
    
    # Train and test the classifier
    for train_indx, test_indx in tqdm(skf.split(X, y)):
        # Get the x_train and y_train
        x_train, y_train = X[train_indx], y[train_indx]

        # Get the x_test and y_test
        x_test, y_test = X[test_indx], y[test_indx]
        
        # Standardize the data
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # Fit the data
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        
        # Get the evaluation metrics
        acc.append(metrics.accuracy_score(y_test, y_pred))
    
    # Save the results
    results = {
        'model_name': params['model_name'],
        'layer_num': params['layer_num'],
        'parameters': {
            'kernel': params['kernel'],
            'probability': params['probability'],
            'C': params['C'],
            'gamma': params['gamma']
        },
        'accuracy': np.mean(acc) * 100,
        'std_accuracy': np.std(acc) * 100
    }
    
    return results

def main():
    # Load the features
    with open(FEAT_PKL_FILE, 'rb') as f:
        feat_data = pickle.load(f)

    # Define the parameter grid
    C_values = [10**i for i in range(0, 4)]  # C from 10^-3 to 10^3
    gamma_values = ['auto', 'scale']
    kernel = 'rbf'
    probability = True

    # Initialize the results list
    all_results = []

    # Iterate over the parameter grid
    for C in C_values:
        for gamma in gamma_values:
            # Create the classifier with the current parameters
            clf = svm.SVC(kernel=kernel, probability=probability, C=C, gamma=gamma)

            # Train the model and get the results
            params = {
                'model_name': model_name,
                'layer_num': layer_num,
                'kernel': kernel,
                'probability': probability,
                'C': C,
                'gamma': gamma
            }
            results = train(feat_data, skf, clf, params)
            all_results.append(results)

    # Load existing results if the file exists
    if os.path.exists(OUTPUT_RESULT_PATH):
        with open(OUTPUT_RESULT_PATH, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    # Append new results to existing results
    existing_results.extend(all_results)

    # Save all results to the JSON file
    with open(OUTPUT_RESULT_PATH, 'w') as f:
        json.dump(existing_results, f, indent=4)

if __name__ == '__main__':
    model_name = sys.argv[1]   
    layer_num = sys.argv[2]    
    task_type = sys.argv[3]    

    # Make folder for saving features
    feature_folder_path = os.path.join("features_finetuned", task_type)
    FEAT_PKL_FILE = os.path.join(feature_folder_path, f"{model_name}_{layer_num}.pkl")

    # Make folder for saving results
    results_folder_path = os.path.join("results2", task_type)
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    # Output JSON file
    OUTPUT_RESULT_PATH = os.path.join(results_folder_path, "results_svm.json")

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    main()
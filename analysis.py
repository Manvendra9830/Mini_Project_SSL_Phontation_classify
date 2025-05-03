import json

# Assuming your JSON data is in a file called 'data.json'
# Replace 'data.json' with your actual file name
with open('results1\\binary\\results_svm.json', 'r') as file:
    data = json.load(file)

# Filter entries with accuracy > 88
high_accuracy_models = [entry for entry in data if entry['accuracy'] >= 85 ]

# Print results
print(f"Found {len(high_accuracy_models)} models with accuracy > 88:")
print("----------------------------------------")
for model in high_accuracy_models:
    print(f"Model: {model['model_name']}")
    print(f"Layer: {model['layer_num']}")
    print(f"Accuracy: {model['accuracy']}")
    print(f"Std Dev: {model['std_accuracy']}")
    print(f"Parameters: {model['parameters']}")
    print("----------------------------------------")

# Optionally save the filtered results to a new JSON file
with open('high_accuracy_models_svm_finetuned.json', 'w') as output_file:
    json.dump(high_accuracy_models, output_file, indent=4)
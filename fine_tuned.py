import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, HubertForSequenceClassification, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
from tqdm import tqdm

# Custom Dataset Class
class AudioDataset(Dataset):
    def __init__(self, root_dirs, labels, processor, sampling_rate=16000, max_length=160000):  # Max length in samples (10s at 16kHz)
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.max_length = max_length  # To truncate or pad to a fixed length
        self.data = []
        self.labels = []

        for root_dir, label in zip(root_dirs, labels):
            for f_name in os.listdir(root_dir):
                wav_file_path = os.path.join(root_dir, f_name)
                self.data.append(wav_file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_path = self.data[idx]
        label = self.labels[idx]
        wav, sr = librosa.load(wav_path, sr=self.sampling_rate)

        if len(wav) > self.max_length:
            wav = wav[:self.max_length]
        else:
            wav = np.pad(wav, (0, self.max_length - len(wav)), mode='constant')

        inputs = self.processor(wav, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.squeeze(0)  # Remove batch dimension
        return {"input_values": input_values, "labels": torch.tensor(label, dtype=torch.long)}

class AudioDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Extract input_values and labels
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad input_values to the longest in the batch
        batch = self.processor.pad(
            {"input_values": input_values},
            padding="longest",
            return_tensors="pt"
        )
        batch["labels"] = torch.stack(labels)
        return batch

def finetune_model(model_checkpoint, train_dirs, train_labels, output_dir, num_labels=3):
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load processor and model
    if "wav2vec2" in model_checkpoint:
        processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)
    elif "hubert" in model_checkpoint:
        processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)
        model = HubertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)
    else:
        raise ValueError("Unsupported model checkpoint")

    dataset = AudioDataset(train_dirs, train_labels, processor)

    #train_size = int(0.8 * len(dataset))
    #val_size = len(dataset) - train_size
    subset_size = 20
    indices = torch.randperm(len(dataset))[:subset_size]
    subset_dataset = torch.utils.data.Subset(dataset, indices)

    train_size = 15
    val_size = 5 
    train_dataset, val_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size])


    data_collator = AudioDataCollator(processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,  
        processing_class=processor,   
    )

    trainer.train()

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

def main():
    model_name = sys.argv[1]  # e.g., wav2vec2B, wav2vec2L, hubert
    task_type = sys.argv[2]   # e.g., binary, multi

    model_checkpoint = {
        'wav2vec2B': 'facebook/wav2vec2-base-960h',
        'wav2vec2L': 'facebook/wav2vec2-large-960h-lv60-self',
        'hubert': 'facebook/hubert-large-ls960-ft'
    }[model_name]

    # Paths configuration
    ROOT_DIR_H = {'binary': r"data\normal"}[task_type]
    ROOT_DIR_B = {'binary': r"data\breathy"}[task_type]
    ROOT_DIR_P = {'binary': r"data\pressed"}[task_type]

    train_dirs = [ROOT_DIR_H, ROOT_DIR_B, ROOT_DIR_P]
    train_labels = [0, 1, 2]  # Labels for normal, breathy, pressed

    # Output directory for fine-tuned model
    output_dir = f"finetuned_models/{model_name}_{task_type}"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Fine-tune the model
    finetune_model(model_checkpoint, train_dirs, train_labels, output_dir)

if __name__ == "__main__":
    main()
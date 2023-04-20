from typing import List
import os

import wandb
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from sklearn.metrics import accuracy_score
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # other pre-trained models like 'roberta-base'


def load_data(filename, tokenizer):
    df = pd.read_csv(filename, sep='\t', header=None, names=['polarity', 'aspect', 'term', 'offset', 'sentence'])
    sentences = df['sentence'].tolist()
    terms = df['term'].tolist()
    polarities = df['polarity'].tolist()

    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)

    return inputs, polarities


class SentimentDataset(Dataset):
    def __init__(self, inputs, polarities):
        self.inputs = inputs
        self.polarities = torch.tensor(polarities, dtype=torch.long)

    def __len__(self):
        return len(self.polarities)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.polarities[idx]
        return item


def create_data_loader(inputs, polarities, batch_size=16):
    polarity_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
    polarities = [polarity_mapping[p] for p in polarities]
    dataset = SentimentDataset(inputs, polarities)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class SentimentClassifier(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

class Classifier:
    def __init__(self, base_model: str = 'bert-base-uncased'):
        self.model = SentimentClassifier(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.label_encoder = LabelEncoder()

    def train(self, train_filename: str, dev_filename: str, device: torch.device, epochs: int = 3, lr: float = 1e-5):
        # Put the model on the specified device
        self.model.to(device)

        # Load data
        print("CURRENT POSITION is : ", os.getcwd())
        print("Looking for : ", os.getcwd()+"/"+ train_filename)
        train_inputs, train_polarities = load_data( os.getcwd()+"/"+ train_filename, self.tokenizer)
        self.label_encoder.fit(train_polarities)

        dev_inputs, dev_polarities = load_data(dev_filename, self.tokenizer)

        # Create DataLoaders
        train_loader = create_data_loader(train_inputs, train_polarities, batch_size=16)
        dev_loader = create_data_loader(dev_inputs, dev_polarities, batch_size=16)

        # Set optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_function = torch.nn.CrossEntropyLoss()

        # Training and evaluation
        for epoch in range(epochs):
            self.model.train()
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(device)
                # Updated line to pass input_ids and attention_mask directly
                outputs = self.model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            self.model.eval()
            dev_preds, dev_true = [], []
            with torch.no_grad():
                for batch in dev_loader:
                    inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
                    labels = batch['labels'].to(device)
                    outputs = self.model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=labels)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1).cpu().tolist()
                    dev_preds.extend(predictions)
                    dev_true.extend(labels.cpu().tolist())

            dev_accuracy = accuracy_score(dev_true, dev_preds)
            wandb.log({
                'epoch': epoch, 
                'dev_accuracy': dev_accuracy,
            })
            print(f'Epoch: {epoch + 1}, Dev Accuracy: {dev_accuracy:.4f}')

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        # Put the model on the specified device
        self.model.to(device)

        data_inputs, data_polarities = load_data(data_filename, self.tokenizer)
        data_loader = create_data_loader(data_inputs, data_polarities, batch_size=16)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in data_loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
                outputs = self.model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).cpu().tolist()
                preds.extend(predictions)

        return self.label_encoder.inverse_transform(preds)


from typing import List

import torch
from typing import Dict, List, Optional
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
    def __init__(self, inputs: Dict[str, torch.Tensor], polarities: Optional[List[int]]):
        self.inputs = inputs
        if polarities is not None:
            self.polarities = torch.tensor(polarities, dtype=torch.long)
        else:
            self.polarities = None

    def __len__(self):
        return len(self.polarities)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.inputs.items()}
        if self.polarities is not None:
            item['labels'] = self.polarities[idx]
        return item


def create_data_loader(inputs: Dict[str, torch.Tensor], polarities: Optional[List[int]] = None, batch_size: int = 16) -> DataLoader:
    if polarities is None:
        polarities = [0] * len(inputs["input_ids"])
    
    dataset = SentimentDataset(inputs, polarities)
    return DataLoader(dataset, batch_size=batch_size)


def create_data_loader_without_labels(inputs, batch_size=16):
    dataset = SentimentDataset(inputs, None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


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

    def train(self, train_filename: str, dev_filename: str, device: torch.device, epochs: int = 10):
        # Put the model on the specified device
        self.model.to(device)

        # Load data
        train_inputs, train_polarities = load_data(train_filename, self.tokenizer)
        dev_inputs, dev_polarities = load_data(dev_filename, self.tokenizer)

        # Transform polarities using label_encoder
        train_polarities_transformed = self.label_encoder.fit_transform(train_polarities)
        dev_polarities_transformed = self.label_encoder.transform(dev_polarities)

        # Create DataLoaders
        train_loader = create_data_loader(train_inputs, train_polarities_transformed, batch_size=16)
        dev_loader = create_data_loader(dev_inputs, dev_polarities_transformed, batch_size=16)

        # Set optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        loss_function = torch.nn.CrossEntropyLoss()

        # Training and evaluation
        for epoch in range(epochs):
            self.model.train()
            loss_list = []
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(device)
                outputs = self.model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=labels)
                loss = outputs.loss
                loss_list.append(loss)
                loss.backward()
                optimizer.step()
            loss_mean = sum(loss_list)/len(loss_list)

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
            print(f'Epoch: {epoch + 1},Loss: {loss_mean:.4f}, Dev Accuracy: {dev_accuracy:.4f}')

    def predict(self, datafile: str, device: str) -> List[int]:
        # Put the model on the specified device
        self.model.to(device)

        data_inputs, _ = load_data(datafile, self.tokenizer)
        data_loader = create_data_loader(data_inputs, None, batch_size=16)

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

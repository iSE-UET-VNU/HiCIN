import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import os
import pandas as pd

class NLIDataset(Dataset):
    def __init__(self, docs, concepts, labels, tokenizer, max_length=512):
        self.docs = docs
        self.concepts = concepts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, idx):
        # [CLS] text [SEP] concept [SEP]
        encoding = self.tokenizer(str(self.docs[idx]),
                                  str(self.concepts[idx]),
                                  add_special_tokens=True,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation="only_first",
                                  return_tensors='pt')
        
        return {
            "input_ids": encoding['input_ids'].squeeze(),
            "attention_mask": encoding['attention_mask'].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
class NLIModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        super(NLIModel, self).__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling of token embeddings
        last_hidden_state = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        logits = self.classifier(self.dropout(pooled_output))
        return logits
    
    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_name": self.model_name,
            "num_labels": self.classifier.out_features,
            "state_dict": self.state_dict()
        }, path)
    
    @staticmethod
    def load(path):
        state = torch.load(path, map_location='cpu')
        model = NLIModel(model_name=state["model_name"], num_labels=state["num_labels"])
        model.load_state_dict(state["state_dict"])
        return model

def train_nli(model, train_loader, val_loader, device, num_epochs=3, lr=2e-5, ckpt_path="models/nli_model.pt"):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            input_ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        val_acc, _ = evaluate_nli(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(ckpt_path)

def evaluate_nli(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            logits = model(input_ids, mask)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds), total_loss / len(loader)

def infer_concept_scores(model, tokenizer, docs, doc_indices, flatten_concepts, flatten_concepts_layers, device, batch_size=32):
    """
    Tracks original document indices to ensure alignment with main training data.
    """
    model.eval()
    all_pairs_docs, all_pairs_concepts, all_pairs_layers, all_pairs_indices = [], [], [], []

    # Build all possible pairs
    for i in range(len(docs)):
        for c_idx, concept in enumerate(flatten_concepts):
            all_pairs_docs.append(docs[i])
            all_pairs_concepts.append(concept)
            all_pairs_layers.append(flatten_concepts_layers[c_idx])
            all_pairs_indices.append(doc_indices[i]) # original index

    infer_ds = NLIDataset(all_pairs_docs, all_pairs_concepts, [0]*len(all_pairs_docs), tokenizer)
    infer_loader = DataLoader(infer_ds, batch_size=batch_size, shuffle=False)

    all_scores = []
    print(f"--- Inferring {len(all_pairs_docs)} pairs ---")
    with torch.no_grad():
        for batch in tqdm(infer_loader):
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            probs = torch.softmax(logits, dim=-1)[:, 1] # Probability of Entailment
            all_scores.extend(probs.cpu().numpy().tolist())

    return pd.DataFrame({
        "doc_idx": all_pairs_indices,
        "concept": all_pairs_concepts,
        "layer": all_pairs_layers,
        "nli_score": all_scores
    })


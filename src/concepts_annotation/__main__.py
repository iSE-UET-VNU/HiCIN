from utils.json_utils import JSONFileReader
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from src.concepts_annotation.weak_label_generator import WeakLabelAnnotator
from src.concepts_annotation.nli_model import NLIModel, NLIDataset, train_nli, infer_concept_scores

def main():
    # 1. Load Configs & Data
    dataset_configs = JSONFileReader.read('configs/dataset_configs.json')
    train_df = pd.read_csv(dataset_configs['train_data_path'])
    
    docs = train_df[dataset_configs.get('text_column', 'text')].tolist()
    labels = train_df[dataset_configs.get('label_column', 'label')].tolist()
    doc_indices = train_df.index.tolist() # original indices
    
    concepts = JSONFileReader.read('configs/concepts.json')
    
    # 2. Weak Labeling
    annotator = WeakLabelAnnotator()
    annotated_df = annotator.annotate(
        docs, labels, doc_indices, 
        concepts['concrete_concepts'], concepts['abstract_concepts']
    )
    
    # 3. Prepare NLI Model
    nli_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Balance and Split NLI dataset
    nli_train_df, nli_val_df = train_test_split(annotated_df, test_size=0.2, stratify=annotated_df['weak_label'])
    
    # sample nli_train_df for quick testing
    # nli_train_df = nli_train_df.sample(20, random_state=42)
    
    train_loader = DataLoader(NLIDataset(nli_train_df['text'].tolist(), 
                                         nli_train_df['concept'].tolist(), 
                                         nli_train_df['weak_label'].tolist(), 
                                         tokenizer), batch_size=8, shuffle=True)
    
    val_loader = DataLoader(NLIDataset(nli_val_df['text'].tolist(), 
                                       nli_val_df['concept'].tolist(), 
                                       nli_val_df['weak_label'].tolist(), 
                                       tokenizer), batch_size=8)

    # 4. Train NLI Model
    nli_model = NLIModel(model_name=nli_model_name).to(device)
    train_nli(nli_model, train_loader, val_loader, device)

    # 5. Full Inference for HiCIN Matrix
    nli_model = NLIModel.load("models/nli_model.pt").to(device)
    
    # Flatten concepts with fixed order
    concrete_list = [c for lst in concepts['concrete_concepts'].values() for c in lst]
    abstract_list = [c["concept"] for lst in concepts['abstract_concepts'].values() for c in lst]
    
    flatten_concepts = concrete_list + abstract_list
    layers = [0]*len(concrete_list) + [1]*len(abstract_list)
    
    scores_df = infer_concept_scores(
        nli_model, tokenizer, docs, doc_indices, 
        flatten_concepts, layers, device
    )
    
    # 6. Save results
    scores_df.to_csv("configs/concept_scores.csv", index=False)
    print(f"[SUCCESS] Concept scores saved. Ready for HiCIN training.")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    main()


import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.hicin_training.hicin import (
    HiCIN, 
    train_hicin, 
    evaluate_hicin,
    prepare_hicin_data
)
from src.hicin_training.text_encoder import TextEncoder
from utils.json_utils import JSONFileReader

def main():
    # 1. Load Configurations
    dataset_configs = JSONFileReader.read('configs/dataset_configs.json')
    text_col = dataset_configs.get('text_column', 'text')
    label_col = dataset_configs.get('label_column', 'label')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Raw Data and Pre-computed NLI Scores
    train_raw_df = pd.read_csv(dataset_configs['train_data_path'])
    test_raw_df = pd.read_csv(dataset_configs.get('test_data_path'))
    scores_df = pd.read_csv("configs/concept_scores.csv").drop_duplicates(subset=['doc_idx', 'concept'])
    
    # Filter only rows that have been processed in the NLI/Concept Annotation step
    scored_indices = scores_df['doc_idx'].unique()
    valid_train_total = train_raw_df[train_raw_df.index.isin(scored_indices)]
    valid_test_df = test_raw_df[test_raw_df.index.isin(scored_indices)]
    
    print(f"[*] Valid Train samples: {len(valid_train_total)}")
    print(f"[*] Valid Test samples:  {len(valid_test_df)}")

    # Split original training data into Train and Validation sets (80/20)
    train_df, val_df = train_test_split(
        valid_train_total, 
        test_size=0.2, 
        stratify=valid_train_total[label_col], 
        random_state=42
    )
    
    # 3. Process Concept Hierarchies from JSON metadata
    concepts_metadata = JSONFileReader.read('configs/concepts.json')
    concrete_mapping = concepts_metadata.get('concrete_concepts', {})
    abstract_mapping = concepts_metadata.get('abstract_concepts', {})
    
    # Flatten concepts for model layers
    label_list = list(concrete_mapping.keys())
    concrete_list = [c for c_list in concrete_mapping.values() for c in c_list]
    abstract_list = [c_obj["concept"] for c_list in abstract_mapping.values() for c_obj in c_list]
    
    # 4. Initialize Text Encoder for global context and InfoNCE alignment
    print(f"\nInitializing TextEncoder with BERT backbone on {device}...")
    text_encoder = TextEncoder("bert-base-uncased", device=device)

    # 5. Prepare CINDatasets
    print("\n[*] Processing Training Dataset...")
    train_ds = prepare_hicin_data(train_df, text_col, label_col, concrete_list, abstract_list, label_list, scores_df, text_encoder)
    
    print("[*] Processing Validation Dataset...")
    val_ds = prepare_hicin_data(val_df, text_col, label_col, concrete_list, abstract_list, label_list, scores_df, text_encoder)
    
    print("[*] Processing Test Dataset...")
    test_ds = prepare_hicin_data(valid_test_df, text_col, label_col, concrete_list, abstract_list, label_list, scores_df, text_encoder)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # 6. Compute Static Semantic Buffers (SBERT priors) for InfoNCE Alignment
    # These represent the "pure" semantic meaning of the concept names
    print("\nComputing static embeddings for concept names...")
    with torch.no_grad():
        s_raw_buffers = {
            'kw': text_encoder.encode(concrete_list).to(device),
            'ab': text_encoder.encode(abstract_list).to(device),
            'lb': text_encoder.encode(label_list).to(device)
        }
    
    # 7. Initialize HiCIN Model
    model = HiCIN(
        num_kw=len(concrete_list),
        num_ab=len(abstract_list),
        num_lb=len(label_list),
        s_dim=text_encoder.get_hidden_size(),
        emb_dim=128,
        attn_dim=128
    )
    
    # 8. Execution: Training
    print(f"\n{'='*45}")
    print(f"HiCIN PIPELINE CONFIGURATION")
    print(f"Train/Val/Test:    {len(train_df)} / {len(val_df)} / {len(valid_test_df)}")
    print(f"Hierarchy (C/A/L): {len(concrete_list)} / {len(abstract_list)} / {len(label_list)}")
    print(f"{'='*45}\n")
    
    model_save_path = "models/best_hicin_model.pt"
    train_hicin(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        s_raw_buffers=s_raw_buffers,
        epochs=300,
        lr=1e-3,
        alpha=0.01,
        save_path=model_save_path
    )

    # 9. --- FINAL TESTING PHASE ---
    print(f"\n{'#'*20} FINAL TEST EVALUATION {'#'*20}")
    
    # Load the best weights discovered during training
    if os.path.exists(model_save_path):
        print(f"Loading best model weights from {model_save_path}...")
        model.load_state_dict(torch.load(model_save_path))
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            z_kw = batch['z_kw'].to(device)
            d_ab = batch['d_ab'].to(device)
            h_text = batch['h_text'].to(device)
            targets = batch['label'].to(device)
            
            output = model(z_kw, d_ab, h_text)
            preds = torch.argmax(output['logits'], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Print Comprehensive Metrics
    print("\n[!] Test Performance Report:")
    print(classification_report(all_labels, all_preds, target_names=label_list))
    
    print("\n[!] Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    print(f"\n[SUCCESS] HiCIN training and testing completed.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()


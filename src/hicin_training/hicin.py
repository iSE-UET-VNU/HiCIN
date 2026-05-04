import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any

class CINDataset(Dataset):
    """
    Dataset for Hierarchical Concept Interaction Network (HiCIN).
    """
    def __init__(
        self, 
        texts: List[str], 
        labels: List[str], 
        label_list: List[str], 
        kw_scores: np.ndarray, 
        ab_scores: np.ndarray, 
        sbert_embeddings: np.ndarray
    ):
        self.texts = texts
        self.labels = labels
        self.label_to_idx = {lb: i for i, lb in enumerate(label_list)}

        self.z_kw = torch.tensor(kw_scores, dtype=torch.float32)
        self.d_ab = torch.tensor(ab_scores, dtype=torch.float32)
        self.h_text = torch.tensor(sbert_embeddings, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label_id = torch.tensor(self.label_to_idx[self.labels[idx]], dtype=torch.long)

        return {
            "z_kw": self.z_kw[idx],     # Concrete/Keyword NLI scores
            "d_ab": self.d_ab[idx],     # Abstract NLI scores
            "h_text": self.h_text[idx], # Global context SBERT embedding
            "label": label_id,
            "text": self.texts[idx]     # Raw text for potential explainability tracing
        }

class HiCIN(nn.Module):
    """
    Hierarchical Concept Interaction Network (HiCIN).
    """
    def __init__(
        self, 
        num_kw: int, 
        num_ab: int, 
        num_lb: int, 
        s_dim: int, 
        emb_dim: int = 128, 
        attn_dim: int = 128
    ):
        super(HiCIN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_kw, self.num_ab, self.num_lb = num_kw, num_ab, num_lb
        
        # 1. Learnable Concept Embeddings (E matrix in paper)
        self.kw_embs = nn.Parameter(torch.randn(num_kw, emb_dim) * 0.02)
        self.ab_embs = nn.Parameter(torch.randn(num_ab, emb_dim) * 0.02)
        self.lb_embs = nn.Parameter(torch.randn(num_lb, emb_dim) * 0.02)
        
        # 2. Semantic Projector (Maps emb_dim to SBERT s_dim for InfoNCE)
        self.infonce_projector = nn.Linear(emb_dim, s_dim, bias=False)
        
        # 3. Hierarchical Interaction Attention Layers
        self.Wq = nn.Linear(emb_dim, attn_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, attn_dim, bias=False)
        self.Uq = nn.Linear(emb_dim, attn_dim, bias=False)
        self.Uk = nn.Linear(emb_dim, attn_dim, bias=False)

        # 4. Contextual Gating Mechanism (Beta)
        self.gate_mlp = nn.Sequential(
            nn.Linear(s_dim + emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.b_ab = nn.Parameter(torch.zeros(num_ab))
        self.b_lb = nn.Parameter(torch.zeros(num_lb))
        self.to(self.device)
    
    def _get_attn(self, Q_embs: torch.Tensor, K_embs: torch.Tensor, mode: str = 'kw_ab') -> torch.Tensor:
        """Computes the inter-layer interaction matrix."""
        Q = self.Wq(Q_embs) if mode == 'kw_ab' else self.Uq(Q_embs)
        K = self.Wk(K_embs) if mode == 'kw_ab' else self.Uk(K_embs)
        scores = (Q @ K.T) / math.sqrt(Q.shape[-1])
        return torch.softmax(scores, dim=0) # Softmax over keys (columns)
    
    def compute_infonce_loss(
        self, 
        s_kw_raw: torch.Tensor, 
        s_ab_raw: torch.Tensor, 
        s_lb_raw: torch.Tensor, 
        tau: float = 0.07
    ) -> torch.Tensor:
        """Aligns learnable concept embeddings with their semantic SBERT priors."""
        E_all = torch.cat([self.kw_embs, self.ab_embs, self.lb_embs], dim=0)
        S_raw = torch.cat([s_kw_raw, s_ab_raw, s_lb_raw], dim=0)
        
        e_proj = F.normalize(self.infonce_projector(E_all), dim=-1)
        s_norm = F.normalize(S_raw, dim=-1)
        
        logits = (e_proj @ s_norm.T) / tau
        targets = torch.arange(logits.shape[0], device=self.device)
        return F.cross_entropy(logits, targets)
    
    def forward(self, z_kw: torch.Tensor, d_ab: torch.Tensor, h_text: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = z_kw.shape[0]

        # 1. Generate Interaction Matrices
        A_kw_ab = self._get_attn(self.kw_embs, self.ab_embs, 'kw_ab')
        A_ab_lb = self._get_attn(self.ab_embs, self.lb_embs, 'ab_lb')

        # 2. Bottom-Up Information Propagation (Concrete -> Abstract)
        r_ab = z_kw @ A_kw_ab + self.b_ab 

        # 3. Contextual Gating 
        # Expand h_text to match abstract concepts dimension
        h_exp = h_text.unsqueeze(1).expand(-1, self.num_ab, -1)
        e_exp = self.ab_embs.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Calculate beta gate for each abstract concept
        beta = self.gate_mlp(torch.cat([h_exp, e_exp], dim=-1)).squeeze(-1)

        # 4. Feature Fusion (Combine NLI direct scores and propagated scores)
        z_ab = beta * d_ab + (1 - beta) * r_ab

        # 5. Final Classification (Abstract -> Label)
        logits = z_ab @ A_ab_lb + self.b_lb
        
        return {"logits": logits, "beta": beta, "A_kw_ab": A_kw_ab}

# --- Utility Functions ---

def prepare_hicin_data(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    concrete_list: List[str], 
    abstract_list: List[str], 
    label_list: List[str], 
    scores_df: pd.DataFrame, 
    encoder: Any
) -> CINDataset:
    """
    Prepares the dataset by aligning pre-computed NLI scores with specific dataframe indices.
    """
    # 1. Pivot scores matrix. Ensure 'doc_idx' matches df.index
    matrix_df = scores_df.pivot_table(
        index='doc_idx', columns='concept', values='nli_score', aggfunc='mean'
    ).fillna(0)
    
    current_indices = df.index.tolist()
    docs = df[text_col].tolist()
    labels = df[label_col].tolist()

    # 2. Reindex strictly based on the provided DataFrame indices
    # This guarantees row 0 of matrix matches row 0 of docs
    z_kw_matrix = matrix_df.reindex(index=current_indices, columns=concrete_list, fill_value=0).values
    d_ab_matrix = matrix_df.reindex(index=current_indices, columns=abstract_list, fill_value=0).values
    
    print(f"[*] Encoding {len(docs)} documents for global SBERT context...")
    h_text_matrix = encoder.encode(docs).cpu().numpy()
    
    return CINDataset(docs, labels, label_list, z_kw_matrix, d_ab_matrix, h_text_matrix)

def train_hicin(
    model: HiCIN, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    s_raw_buffers: Dict[str, torch.Tensor], 
    epochs: int = 10, 
    lr: float = 1e-3, 
    alpha: float = 0.1,
    save_path: str = "models/best_hicin_model.pt"
):
    """Main training loop for HiCIN."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            z_kw = batch['z_kw'].to(model.device)
            d_ab = batch['d_ab'].to(model.device)
            h_text = batch['h_text'].to(model.device)
            labels = batch['label'].to(model.device)

            output = model(z_kw, d_ab, h_text)
            
            # Hybrid Loss: CrossEntropy (Task) + InfoNCE (Semantic Alignment)
            ce_loss = criterion(output['logits'], labels)
            nce_loss = model.compute_infonce_loss(
                s_raw_buffers['kw'], s_raw_buffers['ab'], s_raw_buffers['lb']
            )
            loss = ce_loss + alpha * nce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(ce=f"{ce_loss.item():.4f}", nce=f"{nce_loss.item():.4f}")

        val_acc = evaluate_hicin(model, val_loader)
        print(f"Epoch {epoch+1} | Avg Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            import os
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model with accuracy: {best_acc:.4f}")

@torch.no_grad()
def evaluate_hicin(model: HiCIN, loader: DataLoader) -> float:
    """Evaluates classification accuracy on a given dataloader."""
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        z_kw = batch['z_kw'].to(model.device)
        d_ab = batch['d_ab'].to(model.device)
        h_text = batch['h_text'].to(model.device)
        labels = batch['label'].to(model.device)
        
        out = model(z_kw, d_ab, h_text)
        preds = torch.argmax(out['logits'], dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return correct / total if total > 0 else 0.0


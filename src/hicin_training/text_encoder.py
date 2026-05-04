import torch
import torch.nn as nn
from typing import List, Union
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    """
    A utility class to encode raw text into dense embeddings using 
    HuggingFace Transformer models with mean pooling.
    """
    def __init__(self, model_name: str, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # 1. Initialize Tokenizer and handle missing Pad Tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._fix_tokenizer_padding()

        # 2. Load the Transformer backbone
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # 3. Synchronize vocabulary size if new special tokens were added
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()  # Default to evaluation mode for feature extraction

    def _fix_tokenizer_padding(self):
        """
        Handles models without a default pad_token (e.g., GPT-2).
        Assigns eos_token as pad_token or adds a new [PAD] token.
        """
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def get_hidden_size(self) -> int:
        """Returns the hidden dimension size of the backbone model."""
        return self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]], max_length: int = 512) -> torch.Tensor:
        """
        Encodes raw text strings into representational vectors (B, H).
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize input sequences
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        ).to(self.device)

        # Extract features from the Transformer backbone
        outputs = self.model(**inputs)
        
        if not hasattr(outputs, "last_hidden_state"):
            raise ValueError(f"Model {self.model_name} does not return 'last_hidden_state'.")
            
        last_hidden = outputs.last_hidden_state  # Shape: (B, L, H)
        attention_mask = inputs["attention_mask"]
        
        # Perform Mean Pooling
        embeddings = self.mean_pooling(last_hidden, attention_mask)

        return embeddings

    def mean_pooling(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the average of token embeddings while ignoring padding tokens.
        """
        # Expand mask: [B, L] -> [B, L, 1] then match hidden size [B, L, H]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        
        # Sum embeddings across the sequence length dimension, ignoring pad positions
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        
        # Count actual tokens per sentence (avoid division by zero with clamp)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask
    

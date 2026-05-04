import math
import numpy as np
import pandas as pd
import spacy
from typing import List, Dict, Set, Optional
from collections import defaultdict, Counter
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

class ConcreteConceptGenerator:
    def __init__(self, allowed_pos: Set[str] = None):
        if allowed_pos is None:
            self.allowed_pos = {"NOUN", "VERB", "ADJ"}
        else:
            self.allowed_pos = allowed_pos

    def _extract_tokens(self, text: str) -> List[str]:
        doc = nlp(text)
        return [
            token.lemma_.lower() for token in doc 
            if token.pos_ in self.allowed_pos and not token.is_stop and token.is_alpha
        ]
    
    def generate_concrete_concepts(self, 
                                   documents: List[str], 
                                   labels: List[int], 
                                   tau: float = 0.02, 
                                   min_df_doc: int = 5) -> Dict[int, List[str]]:
        unique_labels = sorted(list(set(labels)))
        num_classes = len(unique_labels)
        
        # Frequency statistics
        doc_freq_by_label = defaultdict(lambda: defaultdict(int))
        label_counts = Counter(labels)
        term_global_df = Counter()
        label_to_candidates = defaultdict(set)

        print("Step 1: Extracting candidates and calculating frequencies...")
        for doc_text, lab in tqdm(zip(documents, labels), total=len(documents)):
            tokens = set(self._extract_tokens(str(doc_text)))
            for t in tokens:
                doc_freq_by_label[lab][t] += 1
                term_global_df[t] += 1
                label_to_candidates[lab].add(t)
        
        # Calculate ILF scores
        ilf_scores = {}
        valid_terms = {t for t, df in term_global_df.items() if df >= min_df_doc}
        
        for term in valid_terms:
            relevant_label_count = 0
            for lab in unique_labels:
                rel_freq = doc_freq_by_label[lab][term] / label_counts[lab]
                if rel_freq > tau:
                    relevant_label_count += 1
            
            if relevant_label_count > 0:
                ilf_scores[term] = math.log(num_classes / relevant_label_count)
            else:
                ilf_scores[term] = 0.0
            
        # Compute temporary DF-ILF scores for ALL (term, label) pairs
        all_scores_by_label = defaultdict(dict)
        for lab in unique_labels:
            for term in label_to_candidates[lab]:
                if term not in valid_terms: continue
                
                norm_df = math.log(1 + (doc_freq_by_label[lab][term] / label_counts[lab]))
                final_score = norm_df * ilf_scores.get(term, 0.0)
                
                if final_score > 0:
                    all_scores_by_label[lab][term] = final_score

        # Best Label for each term
        best_label_for_term = {} 
        for lab, term_scores in all_scores_by_label.items():
            for term, score in term_scores.items():
                if term not in best_label_for_term or score > best_label_for_term[term][1]:
                    best_label_for_term[term] = (lab, score)

        # Group again and perform ranking
        exclusive_terms_per_label = defaultdict(list)
        for term, (best_lab, best_score) in best_label_for_term.items():
            exclusive_terms_per_label[best_lab].append((term, best_score))

        ranked_result = {}
        for lab in unique_labels:
            # Sort terms of each label by descending score
            sorted_terms = sorted(exclusive_terms_per_label[lab], key=lambda x: x[1], reverse=True)
            ranked_result[lab] = [t for t, s in sorted_terms]
            
        return ranked_result


class DynamicConceptSelector:
    def __init__(self, theta: float = 0.01, k: int = 5, metric: str = "macro_f1"):
        self.theta = theta
        self.k = k
        self.metric = metric
    
    def select(self, docs: List[str], labels: List[int], label_candidates: Dict[int, List[str]]) -> Dict[int, List[str]]:
        y = np.array(labels)
        unique_labels = sorted(list(label_candidates.keys()))
        
        # Split train/val 8:2
        idx = np.arange(len(docs))
        tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        
        # Pre-compute TF-IDF matrix for all candidates
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        X_full = vectorizer.fit_transform(docs)
        vocab = vectorizer.vocabulary_
        
        selected_per_label = {lab: [] for lab in unique_labels}
        best_score = 0.0
        prev_score = 0.0
        
        print(f"Step 2: Dynamic Selection with Proxy Model (k={self.k})...")

        max_iters = max(len(c) for c in label_candidates.values()) // self.k + 1

        for it in range(max_iters):
            # Add k new concepts for each label
            current_iter_features = []
            for lab in unique_labels:
                start = it * self.k
                end = (it + 1) * self.k
                new_batch = label_candidates[lab][start:end]
                selected_per_label[lab].extend(new_batch)
            
            # Gather all selected concepts to train the proxy model
            all_selected = list(set([t for sublist in selected_per_label.values() for t in sublist]))
            indices = [vocab[t] for t in all_selected if t in vocab]
            
            if not indices: continue
            
            # Slice matrix và train Proxy (Logistic Regression)
            X_tr, X_va = X_full[tr_idx][:, indices], X_full[va_idx][:, indices]
            clf = LogisticRegression(max_iter=500)
            clf.fit(X_tr, y[tr_idx])
            
            y_pred = clf.predict(X_va)
            if self.metric == "accuracy":
                score = accuracy_score(y[va_idx], y_pred)
            else:
                score = f1_score(y[va_idx], y_pred, average="macro")
            
            # Check improvement condition
            if it == 0:
                best_score = score
            else:
                if score - prev_score < self.theta:
                    # If improvement is insufficient, revert to previous iteration's concept set and stop
                    print(f"Iteration {it}: Gain {score - prev_score:.4f} < {self.theta}. Terminating.")
                    for lab in unique_labels:
                        selected_per_label[lab] = selected_per_label[lab][:-self.k]
                    break
                best_score = score
            
            prev_score = score
            print(f"Iteration {it+1}: Concepts={len(all_selected)}, {self.metric}={score:.4f}")

        return selected_per_label

import pandas as pd
import spacy
import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Set, Any

class WeakLabelAnnotator:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        print(f"Loading Spacy model: {spacy_model}...")
        self.nlp = spacy.load(spacy_model)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    def _get_synonyms(self, word: str) -> List[str]:
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower().replace('_', ' '))
        return list(synonyms)

    def _build_lemmatized_synonym_dict(self, concepts: List[str]) -> Dict[str, Set[str]]:
        synonym_dict = defaultdict(set)
        for kw in concepts:
            kw_lemma = kw.lower()
            synonym_dict[kw_lemma].add(kw_lemma)
            for syn in self._get_synonyms(kw):
                doc = self.nlp(syn)
                if len(doc) > 0:
                    lemma = doc[0].lemma_.lower()
                    synonym_dict[kw_lemma].add(lemma)
        return synonym_dict

    def _contains_concept(self, text_lemmas_set: Set[str], text_lemmas_str: str, keywords_to_check: Set[str]) -> bool:
        for kw in keywords_to_check:
            if " " in kw:
                if kw in text_lemmas_str: return True
            elif kw in text_lemmas_set: return True
        return False

    def annotate(self, 
                 docs: List[str], 
                 labels: List[Any], 
                 doc_indices: List[int],
                 concrete_concepts: Dict[str, List[str]],
                 abstract_concepts: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:

        records = []
        
        # 1. Pre-build synonyms
        all_concrete_kws = [kw for kws in concrete_concepts.values() for kw in kws]
        synonym_map = self._build_lemmatized_synonym_dict(list(set(all_concrete_kws)))

        # 2. Batch processing
        print("Batch processing documents...")
        parsed_docs = list(self.nlp.pipe(tqdm(docs), disable=["ner", "parser"], batch_size=1000))

        print("Generating weak labels...")
        for i in tqdm(range(len(docs))):
            doc_text = docs[i]
            true_label = str(labels[i])
            original_idx = doc_indices[i] # Get original CSV index
            parsed_doc = parsed_docs[i]
            
            doc_lemmas_set = {t.lemma_.lower() for t in parsed_doc if t.is_alpha and not t.is_stop}
            doc_lemmas_str = " ".join([t.lemma_.lower() for t in parsed_doc if t.is_alpha and not t.is_stop])

            active_concrete_map = {}

            # --- LAYER 1: CONCRETE ---
            for c_label, kws in concrete_concepts.items():
                c_label_str = str(c_label)
                for kw in kws:
                    syns = synonym_map.get(kw.lower(), {kw.lower()})
                    is_present = self._contains_concept(doc_lemmas_set, doc_lemmas_str, syns)
                    
                    # Logic: Score = 1 if keyword exists AND label matches document
                    score = 1.0 if (is_present and c_label_str == true_label) else 0.0
                    active_concrete_map[kw.lower()] = score
                    
                    records.append({
                        "doc_idx": original_idx, # original index
                        "text": doc_text,
                        "doc_label": true_label,
                        "layer": "concrete",
                        "concept": kw,
                        "concept_label": c_label_str,
                        "weak_label": score
                    })

            # --- LAYER 2: ABSTRACT ---
            for a_label, a_list in abstract_concepts.items():
                a_label_str = str(a_label)
                for item in a_list:
                    concept_name = item["concept"]
                    support_kws = item["supporting_keywords"]
                    
                    is_any_kw_active = any(active_concrete_map.get(kw.lower(), 0.0) == 1.0 for kw in support_kws)
                    score_abs = 1.0 if is_any_kw_active else 0.0
                    
                    records.append({
                        "doc_idx": original_idx, # original index
                        "text": doc_text,
                        "doc_label": true_label,
                        "layer": "abstract",
                        "concept": concept_name,
                        "concept_label": a_label_str,
                        "weak_label": score_abs
                    })

        return pd.DataFrame(records)


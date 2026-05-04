# A Semantically Constrained Concept Bottleneck Model for Interpretable Text Classification: HiCIN

This paper introduces **HICIN**, a CBM framework that incorporates hierarchical concept modeling and semantically constrained training for interpretable text classification. Specifically, HICIN constructs a hierarchical concept space that integrates data-specific concrete concepts with high-level abstract concepts. This design enables structured reasoning from fine-grained textual evidence to more general semantic representations. To support efficient and reliable concept annotation, HICIN adopts a two-stage strategy that combines lexical-based initialization with semantic refinement using a Natural Language Inference model. Furthermore, HICIN introduces a hierarchical label predictor jointly optimized with classification and semantic alignment objectives, explicitly enforcing concept-based reasoning while mitigating information leakage. Extensive experiments on five benchmark datasets across multiple backbone models show that HICIN consistently outperforms state-of-the-art CBM baselines while requiring fewer concepts. On average, HICIN improves classification performance by up to 103% compared to the baselines.

Paper: .....
---

## 🧩 Architecture & Core Components

*   **Hierarchical Concept Space:** Uses both **concrete concepts** (specific words) and **abstract concepts** (high-level semantic clusters).
*   **Two-Stage Annotation:** Combines **lexical matching** with **Natural Language Inference (NLI)** to refine concept scores based on meaning rather than just keywords.
*   **Semantic-Constrained Training:** Implements a dual-loss function to ensure the model's internal concepts align with their actual semantic definitions.

---

## ⚙️ Installation

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/HiCIN.git
cd HiCIN

# Prepare environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Dataset Configuration

Before running the pipeline, update your configuration in `configs/dataset_configs.json`. This file defines the data paths, label mappings, and task descriptions required for both concept discovery and training.

**Example `configs/dataset_configs.json`:**
```json
{
    "data_name": "ecommerce_sampled",
    "train_data_path": "data/ecommerce_sampled/train.csv",
    "test_data_path": "data/ecommerce_sampled/test.csv",
    "label_column": "label",
    "text_column": "text",
    "label_mapping": {
        "Books": "Books",
        "Clothing & Accessories": "Clothing and Accessories",
        "Household": "Household",
        "Electronics": "Electronics"
    },
    "label_descriptions": {
        "Books": "Texts related to books, novels, literature, or any written publications.",
        "Clothing & Accessories": "Texts describing apparel, fashion items, shoes, or accessories.",
        "Household": "Texts referring to home goods, kitchen items, cleaning supplies, or domestic products.",
        "Electronics": "Texts about electronic devices, gadgets, appliances, or tech-related items."
    },
    "topic": "ecommerce product classification from text descriptions",
    "task_description": "Classifying product descriptions into correct ecommerce product categories based on textual content and context."
}
```

---

## 🚀 Usage

The pipeline consists of three main steps:

### 1. Concept Discovery
```bash
python3 -m src.concepts_generation
```

### 2. Concept Annotation
Generate the concept-label matrix using an NLI model.
```bash
python3 -m src.concepts_annotation
```

### 3. Model Training
Train the classifier with semantic constraints for high interpretability.
```bash
python3 -m src.hicin_training
```

---

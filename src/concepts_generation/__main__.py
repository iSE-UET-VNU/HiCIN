from utils.json_utils import JSONFileReader, JSONFileWriter
import pandas as pd
from src.concepts_generation.concrete_concept_generator import ConcreteConceptGenerator, DynamicConceptSelector
from src.concepts_generation.abstract_concept_generator import AbstractConceptGenerator

from utils.llm_gemini import GeminiLLM
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    dataset_configs = JSONFileReader.read('configs/dataset_configs.json')
    text_col = dataset_configs.get('text_column', 'text')
    label_col = dataset_configs.get('label_column', 'label')
    
    train_df = pd.read_csv(dataset_configs['train_data_path'])
    docs = train_df[text_col].tolist()
    labels = train_df[label_col].tolist()
    
    print("\n", "="*10, "Generating Concrete Concepts", "="*10)

    generator = ConcreteConceptGenerator(allowed_pos={"NOUN", "VERB", "ADJ"})
    candidate_concepts = generator.generate_concrete_concepts(
        documents=docs, 
        labels=labels, 
        tau=0.02,        
        min_df_doc=1
    )
    
    selector = DynamicConceptSelector(
        theta=0.01, 
        k=5, 
        metric="macro_f1"
    )
    concrete_concepts = selector.select(
        docs=docs, 
        labels=labels, 
        label_candidates=candidate_concepts
    )
    
    print("Selected Concrete Concepts per Label:")
    for label, concepts in concrete_concepts.items():
        print(f"Label {label}: {concepts}")
        
    print("\n", "="*10, "Generating Abstract Concepts", "="*10)
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    llm = GeminiLLM(api_key=GEMINI_API_KEY)
    abstract_generator = AbstractConceptGenerator(llm=llm)

    abstract_concepts = abstract_generator.generate_all_abstracts(concrete_concepts, dataset_configs)
    
    print("Saving concepts to configs/concepts.json...")
    print("Number of Concrete Concepts:", sum(len(c) for c in concrete_concepts.values()))
    print("Number of Abstract Concepts:", sum(len(c) for c in abstract_concepts.values()))

    JSONFileWriter.write(
        {
            "concrete_concepts": concrete_concepts,
            "abstract_concepts": abstract_concepts
        },
        "configs/concepts.json"
    )


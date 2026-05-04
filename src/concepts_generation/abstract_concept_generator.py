from pydantic import BaseModel, Field
from typing import List, Dict, Set, Any

class AbstractConceptItem(BaseModel):
    concept: str = Field(description="A short noun phrase representing the abstract concept.")
    supporting_keywords: List[str] = Field(description="List of keywords from the input that support this concept.")

class AbstractConceptList(BaseModel):
    concepts: List[AbstractConceptItem]
    
def generate_prompt_for_label(
    label_name, 
    label_description,
    keywords,
    all_labels,
    taken_concepts: set = None,
    topic="medical",
    task_description="text classification with multi-hop reasoning"):

    keyword_list_str = ", ".join(keywords)
    all_labels_str = ", ".join(all_labels)

    system_prompt = f"""You are assisting in building a multi-layered reasoning system for {topic} {task_description}.
This system performs concept abstraction to enhance explainability and classification performance."""

    base_prompt = f"""You are working on the **second concept layer** — abstracting surface-level keywords into **mid-level abstract concepts**.

---

### OBJECTIVE:

Given:
- A list of raw keywords related to a specific label.
- The label's name and description.
- The list of **all possible labels** in the classification task.

Your job:
- Analyze the extracted keywords and group them into a set of concise, meaningful mid-level concepts.
- You should determine the optimal number of abstract concepts based on the diversity and semantic coverage of the input keywords.
- For each generated concept, map it back to the specific keywords from the input list that support its definition.

Each concept must be:
- More abstract than individual keywords
- More specific than the label name
- Useful in distinguishing this label from others
- Expressed as a **short noun phrase**, not a sentence

---

### INPUTS:

- Target Label: {label_name}
- Label Description: {label_description}
- All Labels: {all_labels_str}
- Extracted Keywords: {keyword_list_str}"""

    restriction_prompt = ""
    if taken_concepts and len(taken_concepts) > 0:
        taken_concepts_str = ", ".join(f'"{c}"' for c in taken_concepts)
        restriction_prompt = f"""
---

### RESTRICTION:

The following concepts have already been generated for other labels. 
**DO NOT** generate any of these exact concept phrases again to ensure class-specific uniqueness:
[ {taken_concepts_str} ]"""

    output_prompt = f"""
---

### OUTPUT:

Return a JSON list of objects. Each object must contain:
- "concept": <noun phrase>
- "supporting_keywords": <list of relevant keywords from the input list used for this abstraction>"""

    return system_prompt, base_prompt + restriction_prompt + output_prompt


class AbstractConceptGenerator:
    """
    Implements Phase 1.2 of HICIN: Abstract Concept Discovery.
    Uses LLM to group concrete concepts into mid-level semantic categories.
    """

    def __init__(self, llm):
        """
        Args:
            llm: An instance of a class inheriting from BaseLLM (e.g., GeminiLLM).
        """
        self.llm = llm
        self.taken_concepts: Set[str] = set()

    def generate_all_abstracts(
        self, 
        label_concrete_dict: Dict[str, List[str]], 
        dataset_config: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Iterates through each label and uses LLM to discover abstract concepts.
        """
        final_abstract_concepts = {}

        all_labels = [str(l) for l in label_concrete_dict.keys()]
        
        label_descriptions = dataset_config.get('label_descriptions', {})
        topic = dataset_config.get('topic', 'general')
        task_desc = dataset_config.get('task_description', 'text classification with multi-hop reasoning')

        for label in label_concrete_dict.keys():
            label_str = str(label)
            print(f"Generating abstract concepts for label: {label_str}...")
            
            system_prompt, full_prompt = self._build_prompt(
                label=label_str,
                description=label_descriptions.get(label_str, "No description provided."),
                keywords=label_concrete_dict[label],
                all_labels=all_labels,
                topic=topic,
                task_description=task_desc
            )

            combined_prompt = f"{system_prompt}\n\n{full_prompt}"
            
            try:
                response = self.llm.call(combined_prompt, response_model=AbstractConceptList)
                
                label_output = []
                for item in response.concepts:
                    concept_data = item.model_dump()
                    label_output.append(concept_data)
                    
                    self.taken_concepts.add(item.concept.strip().lower())
                
                final_abstract_concepts[label_str] = label_output
                print(f"Successfully generated {len(label_output)} concepts for {label_str}.")
                
            except Exception as e:
                print(f"Error generating for label {label_str}: {e}")
                final_abstract_concepts[label_str] = []

        return final_abstract_concepts

    def _build_prompt(self, label, description, keywords, all_labels, topic, task_description):
        return generate_prompt_for_label(
            label_name=label,
            label_description=description,
            keywords=keywords,
            all_labels=all_labels,
            taken_concepts=self.taken_concepts,
            topic=topic,
            task_description=task_description
        )


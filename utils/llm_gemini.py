from utils.llm_base import BaseLLM
from google import genai
from google.genai import types
from pydantic import BaseModel

class GeminiResponse(BaseModel):
    answer: str

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        
    def get_model_name(self) -> str:
        return self.model_name

    def call(self, prompt: str, response_model: BaseModel) -> BaseModel:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Part(text=prompt)],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": response_model.model_json_schema()
            }
        )
        return response_model.model_validate_json(response.text)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import os
    
    # load API key from environment variable
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    llm = GeminiLLM(api_key=GEMINI_API_KEY)
    print(llm.get_model_name())
    response = llm.call("What is the capital of France?", response_model=GeminiResponse)
    print(response.answer)

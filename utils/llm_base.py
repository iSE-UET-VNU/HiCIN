from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import TypeVar, Generic, Any

# Define a TypeVar for structured output validation
T = TypeVar('T', bound=BaseModel)

class BaseLLM(ABC):
    @abstractmethod
    def call(self, prompt: str, response_model: BaseModel) -> T:
        """
        Sends a prompt to the LLM and returns a structured response.

        Args:
            prompt (str): The input text or instruction for the model.
            response_model (type[T]): A Pydantic class defining the expected output schema.

        Returns:
            T: An instance of the response_model containing the LLM's output.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Returns the name/version of the underlying model.
        """
        pass

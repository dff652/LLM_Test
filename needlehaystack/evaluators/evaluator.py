
from abc import ABC, abstractmethod

class Evaluator(ABC):
    CRITERIA: dict[str, str]

    @abstractmethod
    def evaluate_response(self, response: str) -> int: ...
    
    @abstractmethod
    async def evaluate_response_async(self, response: str, true_answer: str, question_asked: str) -> int: ...
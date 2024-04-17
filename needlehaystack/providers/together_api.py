import os
import json
import openai
from pydantic import BaseModel, Field
from ..config.private_config import TOGETHER_API_KEY
import tiktoken
from typing import Optional


class User(BaseModel):
    score: str = Field(description=" score")
    reasoning: str = Field(description="reasoning")
    
    
class TogetherAPI:
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    def __init__(self, 
                 base_url: str = "https://api.together.xyz/v1",
                 model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = '',
                 question_asked: str = '',)
    
        self.api_key = TOGETHER_API_KEY
        self.base_url = base_url
        self.model_name = model_name
        self.model = openai.OpenAI(base_url= self.base_url,
                               api_key= self.api_key)

        
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        
    
    
    async def evaluate_model(self, prompt: str) -> str: 
        
        response = await self.model.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object", "schema": User.model_json_schema()},
            messages = prompt,
            **self.model_kwargs
            )
        
        return response.choices[0].message.content
    
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        
        return [{
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"
            }]
        
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)
    
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])
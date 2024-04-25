import os
import json
import openai
from pydantic import BaseModel, Field
from ..config.private_config import TOGETHER_API_KEY
from ..config.criteria import CRITERIA_NEEDLEHAYSTACK, CRITERIA_EXAM
import tiktoken
from typing import Optional


class User(BaseModel):
    score: str = Field(description=" score")
    reasoning: str = Field(description="reasoning")
    
    
class TogetherAPIEvaluator:
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    def __init__(self, 
                 base_url: str = "https://api.together.xyz/v1",
                 model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = '',
                 question_asked: str = '',):
    
        self.api_key = TOGETHER_API_KEY
        self.base_url = base_url
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.model = openai.OpenAI(base_url= self.base_url,
                               api_key= self.api_key)

        
        
        # self.tokenizer = tiktoken.encoding_for_model(self.model)
        
        self.CRITERIA = CRITERIA_NEEDLEHAYSTACK
        self.true_answer = true_answer
        self.question_asked = question_asked
    
    
    async def evaluate_model(self, prompt: str) -> str: 
        prompt = self.generate_evaluation_prompt(response)
        
        response = await self.model.chat.completions.create(
            model=self.model_name,
            # response_format={"type": "json_object", "schema": User.model_json_schema()},
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
        
    def generate_evaluation_prompt(self,response: str) -> str | list[dict[str, str]]:
        return [
           {"role": "system",
                "content": """
                You are an evaluator AI designed to assess answers provided by another AI model. You will be given a correct answer, a model's response, and specific criteria for evaluation. Your task is to provide a score based on how well the model's response meets the criteria, 
                Keep your assessment concise and objective.
                """
                },
            {"role": "user",
                "content": f"""
                Question:
                --- --- ---
                {self.question_asked}
                Correct Answer:
                --- --- ---
                {self.true_answer}

                Model Response:
                --- --- ---
                {response}
                Evaluation Criteria:
                --- --- ---
                {self.CRITERIA}
                
                result must in dict form and contain reasoning and score.
                示例如下：
                
                {{
                    "score": 1,
                    "reasoning": "原因"
                
                }}
        
                
                """
            }   
        ]
                
       
    async def evaluate_response_async(self,  response: str) -> dict:
        prompt = self.generate_evaluation_prompt(response)
        
        response = await self.model.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object", "schema": User.model_json_schema()},
            messages = prompt,
            **self.model_kwargs
            )
        
        return response.choices[0].message.content
    
    
    async def evaluate_response(self, response: str) -> dict: 
        try:
            prompt = self.generate_evaluation_prompt(response)
            
            response = self.model.chat.completions.create(
                model=self.model_name,
                # response_format={"type": "json_object", "schema": User.model_json_schema()},
                messages = prompt,
                **self.model_kwargs
                )
            print(response)
            eval_result = json.loads(response.choices[0].message.content)
            
            if isinstance(eval_result, str):
                eval_result = json.loads(eval_result)
                return int(eval_result['score']), eval_result['reasoning']
            elif isinstance(eval_result, dict):
                return int(eval_result['score']), eval_result['reasoning']
        except Exception as e:
            print('Error :', e)
            return 0, 'Error decoding json'
    

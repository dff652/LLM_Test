import os
from .evaluator import Evaluator

from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI
from ..config.private_config import OPENAI_API_KEY
from ..config.criteria import CRITERIA_NEEDLEHAYSTACK, CRITERIA_EXAM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import AutoModel, pipeline
import torch
import json

device_map = 'auto'
device = "cuda"

model_dict ={
    "qwen1.5-7B-Chat": "/home/dff652/llm_models/qwen/Qwen1___5-7B-Chat/",
    "qwen1.5-14B": "/home/dff652/llm_models/qwen/Qwen1___5-14B/",
    "qwen1.5-MoE-A2.7B-Chat": "/home/dff652/llm_models/qwen/Qwen1___5-MoE-A2___7B-Chat/",
    "qwen1.5-32B-Chat": "/home/dff652/llm_models/qwen/Qwen1___5-32B-Chat/",
}

class QwenEvaluator(Evaluator):
    
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 1024,
                                      temperature = 0)
    
    def __init__(self,
                 model_name: str = "qwen1.5-7B-Chat",
                 true_answer: str = '',
                 question_asked: str = '',
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked
        
        self.api_key = None
        model_path = model_dict[model_name]
        
        
        self.evaluator = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map=device_map,
                                                          torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.CRITERIA = CRITERIA_EXAM
    
    
    def generate_answer(self,  response: str, true_answer: str, question_asked: str, criteria: str) -> list[dict[str, str]]:
        return [
        {"role": "system",
         "content": """You are an evaluator AI designed to assess answers provided by another AI model. You will be given a correct answer, a model's response, and specific criteria for evaluation. Your task is to provide a score based on how well the model's response meets the criteria, where 1 is the lowest and 10 is the highest score. Keep your assessment concise and objective."""
        },
        {"role": "user",
         "content": f"""
        Question:
        --- --- ---
        {question_asked}
        Correct Answer:
        --- --- ---
        {true_answer}

        Model Response:
        --- --- ---
        {response}
        Evaluation Criteria:
        --- --- ---
        {criteria}

        Based on the above, provide a score  that reflects how well the model's response aligns with the correct answer according to the given criteria. 
        Also, briefly explain the reason for your score. 
        result must in dict form and contain reasoning and score.
        最终输出必须用markdown格式:
        ```json
        result
        ```
        """
                },
     
        ]
    
    async def evaluate_response_async(self,  response: str, true_answer: str, question_asked: str) -> dict:
        prompt = self.generate_answer(question_asked, true_answer, response, self.CRITERIA)
        text = self.tokenizer.apply_chat_template(prompt,
                                                  tokenize=False,
                                                  add_generation_prompt=True,
                                                  **self.model_kwargs
                                                  )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = self.evaluator.generate(model_inputs.input_ids,
                                       max_new_tokens=1024)
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        eval_response =  self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return eval_response
        # try:
        #     eval_result = json.loads(eval_response)
        #     # print('*'*50)
        #     # print(eval_result)
        #     return eval_result
        # except json.JSONDecodeError as e:
        #     print('-'*50)
        #     print(f"response: {response}")
        #     print(f"true_answer: {true_answer}")
        #     print(f"question_asked: {question_asked}")
        #     print('='*50)
        #     print(f"Error decoding json: {e}")
        #     print(f"eval_response: {eval_response}")
        #     return {'reasoning': 'Error decoding json', 'score': 0}
    
    
    def evaluate_response(self, response: str) -> int:
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])

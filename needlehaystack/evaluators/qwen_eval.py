import os
from .evaluator import Evaluator

from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI
from ..config.private_config import OPENAI_API_KEY
from ..config.criteria import CRITERIA_NEEDLEHAYSTACK, CRITERIA_EXAM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import AutoModel, pipeline
import torch

device_map = 'auto'
device = "cuda"

model_dict ={
    "qwen1.5-7B-Chat": "/home/dff652/llm_models/qwen/Qwen1___5-7B-Chat/",
    "qwen1.5-14B": "/home/dff652/llm_models/qwen/Qwen1___5-14B/",
}

class QwenEvaluator(Evaluator):
    
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 1024,
                                      temperature = 0)
    
    def __init__(self,
                 model_name: str = "qwen1.5-7B-Chat",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = None
        model_path = model_dict[model_name]
        
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
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

        Based on the above, provide a score (1-10) that reflects how well the model's response aligns with the correct answer according to the given criteria. 
        Also, briefly explain the reason for your score. 
        Output must contain reasoning and score and both in dict form .
 
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
        
        generated_ids = self.model.generate(model_inputs.input_ids,
                                       max_new_tokens=1024)
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        eval_result =  self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        
        return eval_result

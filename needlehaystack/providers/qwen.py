import os
from operator import itemgetter
from typing import Optional

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate
import tiktoken

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import AutoModel, pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate

from .model import ModelProvider
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
model_dict ={
    "qwen1.5-7B-Chat": "/home/dff652/llm_models/qwen/Qwen1___5-7B-Chat/",
    "qwen1.5-14B": "/home/dff652/llm_models/qwen/Qwen1___5-14B/",
}

# os.environ["CUDA_VISIBLE_DEVICES"]="0，1"

device_map = 'auto'
device = "cuda"
# device_gpu = "cuda:0"
# device_cpu = "cpu"

class Qwen(ModelProvider):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 1024,
                                      temperature = 0)

    def __init__(self,
                 model_name: str = "qwen1.5-7B-Chat",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        
        # api_key = os.getenv('NIAH_MODEL_API_KEY')
        # if (not api_key):
        #     raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = None
        model_path = model_dict[model_name]
        
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map=device_map,
                                                          torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    async def evaluate_model(self, prompt: str) -> str:
        # 使用 await 等待异步生成结果
        text = self.tokenizer.apply_chat_template(prompt,
                                                  tokenize=False,
                                                  add_generation_prompt=True,
                                                  **self.model_kwargs
                                                  )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = self.model.generate(model_inputs.input_ids,
                                       max_new_tokens=1024)
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        response =  self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        
        return response
    
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
        
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        
        return self.tokenizer.decode(tokens[:context_length])
        
    def get_langchain_runnable(self, context: str) -> str:
        
        template = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        \n ------- \n 
        {context} 
        \n ------- \n
        Here is the user question: \n --- --- --- \n {question} \n Give an correct answer and specific explanation ."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        pipe = pipeline("text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        temperature = 0,
                        max_length=512,
                        top_p=1,
                        repetition_penalty=1.15
                        )
        qwen_model = HuggingFacePipeline(pipeline=pipe)
        
        
        # model = ChatOpenAI(temperature=0, model=self.model_name)
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt 
                | qwen_model 
                )
        return chain
    
    def generate_answer(self, retrieval_question: str) ->  str | list[dict[str, str]]:
        
        return [{
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Give an correct answer and specific explanation"
            }]
        
        
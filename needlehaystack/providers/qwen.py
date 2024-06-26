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
    "qwen1.5-7B-Chat": "/home/data1/llm_models/qwen/Qwen1___5-7B-Chat/",
    "qwen1.5-14B": "/home/data1/llm_models/qwen/Qwen1___5-14B/",
    "qwen1.5-MoE-A2.7B-Chat": "/home/data1/llm_models/qwen/Qwen1___5-MoE-A2___7B-Chat/",
    "qwen1.5-32B-Chat": "/home/data1/llm_models/qwen/Qwen1___5-32B-Chat/",
    "qwen1.5-32B-Chat-AWQ": "/home/data1/llm_models/qwen/Qwen1___5-32B-Chat-AWQ/",
    "qwen1.5-14B-Chat": "/home/data1/llm_models/qwen/Qwen1___5-14B-Chat/",
    "qwen1.5-110B-Chat-AWQ":"/home/data1/llm_models/qwen/Qwen1___5-110B-Chat-AWQ/"
}

# os.environ["CUDA_VISIBLE_DEVICES"]="0，1"

device_map = 'auto'
device = "cuda"
# device_gpu = "cuda:0"
# device_cpu = "cpu"
torch_dtype = torch.float16
# torch_dtype = 'auto'
class Qwen(ModelProvider):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 512,
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
        
        if model_name == 'qwen1.5-MoE-A2.7B-Chat':
            from modelscope import AutoModelForCausalLM, AutoTokenizer
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map=device_map,
                                                          torch_dtype=torch_dtype)
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
    
    # def generate_markdown_answer(self, retrieval_question: str) ->  str | list[dict[str, str]]:
        
    #     return [
    #         {
    #             "role": "system",
    #             "content": """You are a helpful AI bot that answers questions for a user. Keep your response short and direct"""
    #         },
    #         {"role": "user",
    #          "content": f"""{retrieval_question} 
    #         result must in dict form and contain option and explanation.
    #         最终输出必须用markdown格式,示例如下：
    #         ```json
    #             {"option": "a",
    #             "explanation": "解释"}
    #         ```
    #         """
    #         },  
    #     ]
    
    # def generate_answer(self, retrieval_question: str) ->  str | list[dict[str, str]]:
        
    #     return [
    #         {
    #         "role": "system",
    #         "content": """You are a helpful AI bot that answers questions for a user. Please respond with a direct answer followed by a brief explanation, 
    #         both enclosed in a JSON object formatted within a Markdown code block. Incorrect formats will not be accepted."""
    #         },
    #         {"role": "user",
    #          "content": f"""
    #          {retrieval_question} 
    #         result must in dict form and contain option and explanation.
    #         for example:{{"option": "a","explanation": "This is why option A is correct."}}
    #         """
    #         }, 
    #     ]
        
    
    def generate_markdown_answer(self, retrieval_question: str) ->  str | list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Please respond with a direct answer followed by a brief explanation, both enclosed in a JSON object formatted within a Markdown code block. Incorrect formats will not be accepted."
            },
            {"role": "user",
            "content": f"{retrieval_question}\n\nPlease provide your answer in the following format:\n```json\n{{\"option\": \"a\", \"explanation\": \"This is why option A is correct.\"}}\n```\n"
            },  
        ]
        
    
    def generate_answer(self, retrieval_question: str) ->  str | list[dict[str, str]]:
        
        return [
            {
            "role": "system",
            "content": """You are a helpful AI bot that answers questions for a user. Please respond with a direct answer followed by a brief explanation, 
            both enclosed in a JSON object formatted within a Markdown code block. Incorrect formats will not be accepted."""
            },
            {"role": "user",
             "content": f"{retrieval_question}\n\nPlease provide your answer in the following format:\n{{\"option\": \"a\", \"explanation\": \"This is why option A is correct.\"}}\n"
            },  
        ]


        
        
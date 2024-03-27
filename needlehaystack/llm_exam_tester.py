import asyncio
import glob
import json
import os
import time
import re
import numpy as np

from .evaluators import Evaluator
from .providers import ModelProvider

from asyncio import Semaphore
from datetime import datetime, timezone
from .llm_needle_haystack_tester import LLMNeedleHaystackTester
from .llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester

class LLMExamTester(LLMMultiNeedleHaystackTester):
    """
    这个类不使用needlehaystack的方法，而是直接让模型回答问题再评估
    """
    
    def __init__(self,
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None,
                 question = None, 
                 question_dir = "",):
        
        self.evaluator = evaluator
        self.model_to_test = model_to_test
        self.question = question
        self.question_dir = question_dir
        
        
        
        pass
    
    
                
    async def evaluate_and_log(self):
        """
        不需要context直接让模型回答问题再评估
        """
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return
            
        
        prompt = self.model_to_test.generate_answer(self.retrieval_question)
        
        test_start_time = time.time()
        
        response = await self.model_to_test.evaluate_model(prompt)

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Compare the reponse to the actual needle you placed
        score = self.evaluation_model.evaluate_response(response)
        
    
    async def bound_evaluate_and_log(self, sem, *args):
            async with sem:
                await self.evaluate_and_log(*args)    
    
    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

       
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
            
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Exam Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Question: {self.question}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())
        
        
        
        
        
        
import asyncio
import glob
import json
import os
import time
import re
import numpy as np
import pandas as pd
from .evaluators import Evaluator
from .providers import ModelProvider

from concurrent.futures import ProcessPoolExecutor
from asyncio import Semaphore
from datetime import datetime, timezone

class LLMEvaluator():
    def __init__(self,
                 evaluator: Evaluator = None,
                 read_results_path: str = None,
                 save_results_path: str = None,
                 print_ongoing_status = True,
                 num_concurrent_requests = 1
                 ):
        self.evaluation_model = evaluator  
        self.print_ongoing_status = print_ongoing_status
        self.num_concurrent_requests = num_concurrent_requests
        self.eval_results_file_path = read_results_path
        self.model_name = self.evaluation_model.model_name
        
        
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.context_file_location = f'{self.model_name.replace(".", "_")}_st_{start_time_str}'
        
        base_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
        
        if read_results_path is None:
            self.read_results_path = os.path.join(parent_dir, 'exam_results/')
        if save_results_path is None:
            self.save_results_path = os.path.join(parent_dir, 'eval_results/')
        
    
    async def score_answer(self,response, true_answer, question):
        score = await self.evaluation_model.evaluate_response_async(response, true_answer, question)
        return score
    
    async def evaluate_scores_and_update(self, sem, row):
        async with sem:
            test_start_time = time.time()
            model_response = row['model_response']
            true_answer = row['true_answer']
            question = row['instruction']
            score = await self.score_answer(model_response, true_answer, question)
            
            test_end_time = time.time()
            test_eval_time = test_end_time - test_start_time
            print(f"test_eval_time: {test_eval_time}")
            row['score'] = score
            row['eval_time'] = test_eval_time
    
    def scored_file_exists(self, test_file_name):
        # test_file_name_lower = test_file_name.lower()
        for filename in os.listdir(self.save_results_path):
            if filename.endswith('.csv'):
                print(f"filename: {filename}")
                if test_file_name in filename:
                    print(f"Results already scored for {test_file_name}")
                    return True
                else:
                    print(f"Results not scored for {test_file_name}")
            else:
                continue
        return False
    
    async def run_eval_async(self):
        sem = Semaphore(self.num_concurrent_requests)
        
        if not os.path.exists(self.save_results_path):
            os.makedirs(self.save_results_path)
        
        if os.path.isdir(self.read_results_path):
            csv_files = glob.glob(os.path.join(self.read_results_path, '*.csv'))
            for csv_file in csv_files:
                file_name = os.path.basename(csv_file)
                print(f"Processing {csv_file}")
                # 为每个CSV文件处理评分
                if csv_file.endswith('_perf.csv'):
                    print(f"Skipping {csv_file}")
                    continue
                
                if self.scored_file_exists(file_name):
                    print(f"Results already scored for {csv_file}")
                    continue
                
                df = pd.read_csv(csv_file)
                # 定义每个CSV文件的保存路径
                
                
                tasks = [asyncio.create_task(self.evaluate_scores_and_update(sem, row)) for index, row in df.iterrows()]
                await asyncio.gather(*tasks)
                
                save_results_file_path = os.path.join(self.save_results_path, f'{self.context_file_location}_{file_name}')
                df.to_csv(save_results_file_path, index=False)
                print(f"Results saved to {save_results_file_path}")
                        
                
        elif os.path.isfile(self.read_results_path):
            # 如果是单个文件，和之前的逻辑一样
            df = pd.read_csv(self.read_results_path)
            tasks = [asyncio.create_task(self.evaluate_scores_and_update(sem, row)) for index, row in df.iterrows()]
            await asyncio.gather(*tasks)
            save_results_file_path = os.path.join(self.save_results_path, f'{self.context_file_location}.csv') 
            df.to_csv(save_results_file_path, index=False)
            print(f"Results saved to {save_results_file_path}")
        else:
            print("Provided read_results_path is not valid.")
        
           
        
    def print_start_eval_summary(self):
        print ("\n")
        print ("Starting Exam Evaluation...")
        print (f"- Evaluation_model: {self.evaluation_model}")
        print ("\n\n")

    def start_eval(self):
        if self.print_ongoing_status:
            self.print_start_eval_summary()
        asyncio.run(self.run_eval_async())
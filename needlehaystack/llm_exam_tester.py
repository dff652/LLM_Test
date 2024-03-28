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
from .llm_needle_haystack_tester import LLMNeedleHaystackTester
from .llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester

class LLMExamTester(LLMNeedleHaystackTester):
    """
    这个类不使用needlehaystack的方法，而是直接让模型回答问题再评估
    """
    
    def __init__(self,
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None,
                 question = None, 
                 question_type = "exam",
                 question_dir = "/Exam",
                 exam_results_dir = "",
                 exam_set = "exam",
                 print_ongoing_status = True,
                 num_concurrent_requests = 1,
                 save_results = True,
                 frac = 0.25,
                 results_version = 1,
                 ):
        
        self.evaluation_model = evaluator
        self.model_to_test = model_to_test
        self.question = question
        self.question_dir = question_dir
        self.exam_results_dir = exam_results_dir
        self.question_type = question_type
        self.exam_set = exam_set
        self.print_ongoing_status = print_ongoing_status
        self.model_name = self.model_to_test.model_name
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.frac = frac
        self.results_version = results_version
        self.testing_results = []
        
        # 任务开始前设置文件名，包括时间
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.context_file_location = f'{self.model_name.replace(".", "_")}_question_type_{self.question_type}_{start_time_str}_{self.frac}'
                
    async def evaluate_and_log_async(self, question, question_type, ture_answer):
        """
        不需要context直接让模型回答问题再评估
        """
        question_tokens = len(self.model_to_test.encode_text_to_tokens(question))
        
        if self.save_results:
            if self.result_exists(question_type, question_tokens):
                return
            
        
        prompt = self.model_to_test.generate_answer(question)
        print(f"prompt: {prompt}")
        test_start_time = time.time()
        
        response = await self.model_to_test.evaluate_model(prompt)
        
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        print(f"response: {response}")
        print(f"test_elapsed_time: {test_elapsed_time}")
        # Compare the reponse to the actual needle you placed
        score = await self.evaluation_model.evaluate_response_async(response, self.ture_answer, question)
        
        
        answer_tokens = len(self.model_to_test.encode_text_to_tokens(response))
        
        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_name,
            'answer_tokens' : answer_tokens,
            'question_tokens' : question_tokens,
            'question_type' : question_type,
            'version' : self.results_version,
            'ture_answer' : ture_answer,
            'model_response' : response,
            'instruction' : question,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
            }
        
        self.testing_results.append(results)
        
        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"answer_tokens: {answer_tokens} tokens")
            print (f"Score: {score}")
            print (f"Response: {response}\n")
            
        # context_file_location = f'{self.model_name.replace(".", "_")}_question_type_{self.question_type}'    
        
        base_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
        if self.save_results:
            if self.exam_results_dir == "":
                results_dir  = os.path.join(parent_dir, 'exam_results/')
            # Save the context to file for retesting
                
            else:
                results_dir = self.exam_results_dir
            
            if not os.path.exists(results_dir ):
                    os.makedirs(results_dir )
                    
            print({"results_dir ":results_dir })
            results_file_path = os.path.join(results_dir, f'{self.context_file_location}.csv')
            df = pd.DataFrame([results])
            # df_existing = pd.read_csv(results_file_path)
            if not os.path.isfile(results_file_path):
                df.to_csv(results_file_path, mode='w', index=False)
            else:
                # if not df_existing.append(df, ignore_index=True).duplicated().any():
                df.to_csv(results_file_path, mode='a', header=False, index=False)
                print("New record added.")
            
    
            print(f"Results saved to {results_file_path}")
            
            # with open(f'{results_dir}{context_file_location}_results.json', 'w') as f:
            #     json.dump(self.testing_results, f)
                
    
    def result_exists(self, question_type, question_tokens):
        """
        Checks to see if a result has already been evaluated or not
        """
        base_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
        results_dir = os.path.join(parent_dir, 'exam_results/')
        # results_dir = 'results/'
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.csv'):
                
                df = pd.read_csv(os.path.join(results_dir, filename))
                for index, row in df.iterrows():
                    question_type_met = row['question_type'] == question_type
                    question_tokens_met = row['question_tokens'] == question_tokens
                    version_met = row['version'] == self.results_version  # 注意这里已经直接取得version列
                    model_met = row['model'] == self.model_name
                    if question_type_met and question_tokens_met and version_met and model_met:
                        return True
                    
                
            # elif filename.endswith('.json'):
            #     with open(os.path.join(results_dir, filename), 'r') as f:
            #         results = json.load(f)
            #         for result in results:
            #             question_type_met = result['question_type'] == question_type
            #             question_tokens_met = result['question_tokens'] == question_tokens
            #             version_met = result.get('version', 1) == self.results_version
            #             model_met = result['model'] == self.model_name
            #             if question_type_met and question_tokens_met and version_met and model_met:
            #                 return True
        return False

        
    
    async def bound_evaluate_and_log_async(self, sem, *args):
            async with sem:
                await self.evaluate_and_log_async(*args)    

    
    async def run_test_async(self):
        sem = Semaphore(self.num_concurrent_requests)

        base_dir = os.path.abspath(os.path.dirname(__file__))
        tasks = []
        for question_file in glob.glob(os.path.join(base_dir, self.question_dir,'*')):
            
            if question_file.endswith(".xlsx"):
                df_questions = pd.read_excel(question_file)
                questions = df_questions['instruction']
            elif question_file.endswith(".json"):
                with open(question_file, 'r') as f:
                    questions = json.load(f)
            else:
                print(f"Skipping file {question_file}")
                continue
            
            # 随机抽取一部分数据，这里以抽取25%为例
            df_sampled = df_questions.sample(frac=self.frac, random_state=1)
            
            for i in df_sampled.index:
                self.question = df_sampled['instruction'][i]
                self.question_type = df_sampled['kind'][i]
                self.ture_answer = df_sampled['参考'][i]
                
                task = self.bound_evaluate_and_log_async(sem, self.question, self.question_type, self.ture_answer)
                tasks.append(task)
        

        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # task = self.bound_evaluate_and_log(sem)
        # await asyncio.gather(task)
        
    async def evaluate_and_log_wrapper(self, args):
        """为了在进程池中调用evaluate_and_log, 我们需要一个wrapper函数."""
        # 这里解包参数，因为ProcessPoolExecutor.map()传入的参数需要是可迭代的
        question, question_type, true_answer = args
        return await asyncio.get_event_loop().run_in_executor(None, self.evaluate_and_log, question, question_type, true_answer)
    
    
    def evaluate_and_log(self, question, question_type, ture_answer):
        """
        不需要context直接让模型回答问题再评估
        """
        question_tokens = len(self.model_to_test.encode_text_to_tokens(question))
        
        if self.save_results:
            if self.result_exists(question_type, question_tokens):
                return
            
        
        prompt = self.model_to_test.generate_answer(question)
        print(f"prompt: {prompt}")
        test_start_time = time.time()
        
        response = self.model_to_test.evaluate_model(prompt)
        
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        print(f"response: {response}")
        print(f"test_elapsed_time: {test_elapsed_time}")
        # Compare the reponse to the actual needle you placed
        score = self.evaluation_model.evaluate_response_async(response, self.ture_answer, question)
        
        
        answer_tokens = len(self.model_to_test.encode_text_to_tokens(response))
        
        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_name,
            'answer_tokens' : answer_tokens,
            'question_tokens' : question_tokens,
            'question_type' : question_type,
            'version' : self.results_version,
            'ture_answer' : ture_answer,
            'model_response' : response,
            'instruction' : question,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
            }
        
        self.testing_results.append(results)
        
        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"answer_tokens: {answer_tokens} tokens")
            print (f"Score: {score}")
            print (f"Response: {response}\n")
            
        context_file_location = f'{self.model_name.replace(".", "_")}_question_type_{self.question_type}_length_{question_tokens}'    
        
        base_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
        if self.save_results:
            if self.exam_results_dir == "":
                results_dir  = os.path.join(parent_dir, 'exam_results/')
            # Save the context to file for retesting
                
            else:
                results_dir = self.exam_results_dir
            
            if not os.path.exists(results_dir ):
                    os.makedirs(results_dir )
                    
            print({"results_dir ":results_dir })

            with open(f'{results_dir}{context_file_location}_results.json', 'w') as f:
                json.dump(self.testing_results, f)
    
    async def run_test(self):
        sem = asyncio.Semaphore(self.num_concurrent_requests)

        base_dir = os.path.abspath(os.path.dirname(__file__))
        tasks = []
        questions_to_process = []

        # 准备数据
        for question_file in glob.glob(os.path.join(base_dir, self.question_dir, '*')):
            if question_file.endswith(".xlsx"):
                df_questions = pd.read_excel(question_file)
            elif question_file.endswith(".json"):
                with open(question_file, 'r') as f:
                    df_questions = pd.DataFrame(json.load(f))
            else:
                print(f"Skipping file {question_file}")
                continue
            
            # 随机抽取一部分数据，这里以抽取25%为例
            df_sampled = df_questions.sample(frac=self.frac, random_state=1)
        
            for index, row in df_sampled.iterrows():
                questions_to_process.append((row['instruction'], row['kind'], row['参考']))

        # 使用ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        loop = asyncio.get_running_loop()

        for question_data in questions_to_process:
            task = loop.run_in_executor(executor, self.evaluate_and_log, *question_data)
            tasks.append(task)

        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        # 关闭executor
        executor.shutdown(wait=True)
            
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Exam Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Question: {self.question}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test_async())
        
        # TODO 本地模型评估的多进程版本
        # asyncio.run(self.run_test())
        
        
        
        
        
        
        
        
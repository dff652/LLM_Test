import asyncio
import glob
import json
import os
import time
from asyncio import Semaphore
from datetime import datetime, timezone

import numpy as np

from .evaluators import Evaluator
from .llm_needle_haystack_tester import LLMNeedleHaystackTester
from .providers import ModelProvider
from itertools import product

from zoneinfo import ZoneInfo
# 设置为北京时区
beijing_timezone = ZoneInfo("Asia/Shanghai")

class LLMMultiNeedleHaystackTester(LLMNeedleHaystackTester):
    """
    Extends LLMNeedleHaystackTester to support testing with multiple needles in the haystack.
    
    Attributes:
        needles (list): A list of needles (facts) to insert into the haystack (context).
        model_to_test (ModelProvider): The model being tested.
        evaluator (Evaluator): The evaluator used to assess the model's performance.
        print_ongoing_status (bool): Flag to print ongoing status messages.
        eval_set (str): The evaluation set identifier.
    """
    def __init__(self, *args, 
                 needles=[], 
                 retrieval_question = None,
                 haystack_dir="PaulGrahamEssays",
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None, 
                 print_ongoing_status = True,
                 eval_set = "multi-needle-eval-sf",
                 multi_needles = 1,
                 key_word = '',
                 **kwargs):

        super().__init__(*args, 
                         model_to_test=model_to_test,
                         needle=needles[0],
                         haystack_dir=haystack_dir, 
                         retrieval_question=retrieval_question,
                         multi_needles=multi_needles,
                         length_of_needles = len(needles),
                         key_word=key_word,
                         **kwargs) 
                
        self.needles = needles
        self.evaluator = evaluator
        self.model_to_test = model_to_test                  
        self.eval_set = eval_set
        self.model_name = self.model_to_test.model_name
        self.print_ongoing_status = print_ongoing_status
        self.insertion_percentages = []
        self.retrieval_question = retrieval_question
        self.haystack_dir = haystack_dir
        # self.multi_needles = multi_needles
        self.key_word_word = key_word
        
        if len(needles) > 1:            
            self.multi_needles = 1
        else:
            self.multi_needles = 0

    async def insert_needles(self, context, depth_percent, context_length):
        """
        Inserts multiple needles (specific facts or pieces of information) into the original context string at 
        designated depth percentages, effectively distributing these needles throughout the context. This method 
        is designed to test a model's ability to retrieve specific information (needles) from a larger body of text 
        (haystack) based on the placement depth of these needles.

        The method first encodes the context and each needle into tokens to calculate their lengths in tokens. 
        It then adjusts the context length to accommodate the final buffer length. This is crucial for ensuring 
        that the total token count (context plus needles) does not exceed the maximum allowable context length, 
        which might otherwise lead to information being truncated.

        This approach calculates the initial insertion point for the first needle as before but then calculates even 
        spacing for the remaining needles based on the remaining context length. It ensures that needles are 
        distributed as evenly as possible throughout the context after the first insertion. 
        
        Args:
            context (str): The original context string.
            depth_percent (float): The depth percent at which to insert the needles.
            context_length (int): The total length of the context in tokens, adjusted for final buffer.
        
        Returns:
            str: The new context with needles inserted.
        """
        tokens_context = self.model_to_test.encode_text_to_tokens(context)
        context_length -= self.final_context_length_buffer

        # Calculate the total length of all needles in tokens
        total_needles_length = sum(len(self.model_to_test.encode_text_to_tokens(needle)) for needle in self.needles)

        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]
        
        # To evenly distribute the needles, we calculate the intervals they need to be inserted.
        depth_percent_interval = (100 - depth_percent) / len(self.needles)
        
        # Reset the insertion percentages list for the current context
        self.insertion_percentages = []

        # Insert needles at calculated points
        for needle in self.needles:

            tokens_needle = self.model_to_test.encode_text_to_tokens(needle)

            if depth_percent == 100:
                # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
                tokens_context = tokens_context + tokens_needle
            else:
                # Go get the position (in terms of tokens) to insert your needle
                insertion_point = int(len(tokens_context) * (depth_percent / 100))

                # tokens_new_context represents the tokens before the needle
                tokens_new_context = tokens_context[:insertion_point]

                # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
                period_tokens = self.model_to_test.encode_text_to_tokens('.')
                
                # Then we iteration backwards until we find the first period
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]
                    
                # Insert the needle into the context at the found position
                tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]

                # Log 
                insertion_percentage = (insertion_point / len(tokens_context)) * 100
                self.insertion_percentages.append(insertion_percentage)
                print(f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, total length now: {len(tokens_context)} tokens")
                
                # Adjust depth for next needle
                depth_percent += depth_percent_interval  

        new_context = self.model_to_test.decode_tokens(tokens_context)
        return new_context

    def encode_and_trim(self, context, context_length):
        """
        Encodes the context to tokens and trims it to the specified length.
        
        Args:
            context (str): The context to encode and trim.
            context_length (int): The desired length of the context in tokens.
        
        Returns:
            str: The encoded and trimmed context.
        """
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(tokens, context_length)
        return context

    async def generate_context(self, context_length, depth_percent):
        """
        Generates a context of a specified length and inserts needles at given depth percentages.
        
        Args:
            context_length (int): The total length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        
        Returns:
            str: The context with needles inserted.
        """
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = await self.insert_needles(context, depth_percent, context_length)
        return context
    
    async def evaluate_and_log(self, context_length, depth_percent):
        """
        Evaluates the model's performance with the generated context and logs the results.
        
        Args:
            context_length (int): The length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        """
        
        if '/' in self.evaluator.model_name:
            parts = self.evaluator.model_name.split('/')
        # 获取 '/' 后面的部分，即分割后列表的第二个元素
            model_specific_part = parts[1]
        else:
            model_specific_part = self.evaluator.model_name
            
        # needle语言判断
        needle_language = self.language_detection(self.needle)
        
        
        context = self.read_context_files()
        # 上下文语言判断
        context_language = self.language_detection(context)
        
        
        
        
        if self.save_results:
            if self.result_exists(context_length, depth_percent,model_specific_part, context_language, needle_language):
                print(f"Result exists for context_length={context_length}, depth_percent={depth_percent}, model_specific_part={model_specific_part} ,context_language= {context_language}, needle_language={needle_language}")
                return 

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        test_start_time = time.time()

        # LangSmith
        ## TODO: Support for other evaluators 
        if self.evaluator.__class__.__name__ == "LangSmithEvaluator":  
            print("EVALUATOR: LANGSMITH")
            chain = self.model_to_test.get_langchain_runnable(context)
            self.evaluator.evaluate_chain(chain, context_length, depth_percent, self.model_to_test.model_name, self.eval_set, len(self.needles), self.needles, self.insertion_percentages)
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

        else:
            print(f"EVALUATOR: {self.evaluator.model_name}")
            # Prepare your message to send to the model you're going to evaluate
            prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)
            # Go see if the model can answer the question to pull out your random fact
            response = await self.model_to_test.evaluate_model(prompt)
            # Compare the reponse to the actual needle you placed
            # score = self.evaluator.evaluate_response(response)
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time
            
            score, reasoning = await self.evaluator.evaluate_response(response)

            eval_end_time = time.time()
            eval_elapsed_time = eval_end_time - test_end_time
            

            # results = {
            # # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            # 'model' : self.model_to_test.model_name,
            # 'context_length' : int(context_length),
            # 'depth_percent' : float(depth_percent),
            # 'version' : self.results_version,
            # 'needle' : self.needle,
            # 'model_response' : response,
            # 'score' : score,
            # 'test_duration_seconds' : test_elapsed_time,
            # 'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
            # }
            
            results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'question' : self.retrieval_question,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'reasoning' : reasoning,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(beijing_timezone).strftime('%Y-%m-%d %H:%M:%S%z'),
            'evaluator' : model_specific_part,
            'eval_duration_seconds' : eval_elapsed_time,
            'haystack_dir' : self.haystack_dir,
            'context_language' : context_language,
            'needle_language' : needle_language,
            'multi_needles' : self.multi_needles,
            'length_of_needles' : len(self.needles),
        }

            self.testing_results.append(results)

            if self.print_ongoing_status:
                print (f"\n-- Test Summary -- ")
                print (f"Model: {self.model_name}")
                print (f"Duration: {test_elapsed_time:.1f} seconds")
                print (f"Context: {context_length} tokens")
                print (f"Depth: {depth_percent}%")
                print (f"Score: {score}")
                print (f"Reasoning: {reasoning}")
                print (f"Response: {response}")
                print(f"Eval_duration_seconds: {eval_elapsed_time:.1f} seconds\n")

            # context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'
            
            context_file_location = f'{self.model_name.replace(".", "_")}_{context_language}_{self.start_time_str}_{model_specific_part}_{needle_language}'
            base_dir = os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
            
            if self.save_contexts:
                results['file_name'] = context_file_location

                contexts_dir = os.path.join(parent_dir, self.context_dir, f'context_{context_language}_{needle_language}_multi_needles/')
                print({"contexts_dir":contexts_dir})
            
                # Save the context to file for retesting
                if not os.path.exists(contexts_dir):
                    os.makedirs(contexts_dir)

                with open(f'{contexts_dir}_context_len_{context_length}_depth_{int(depth_percent)}%_{context_file_location}.txt', 'w') as f:
                    f.write(context)
                
            if self.save_results:
                results_dir = os.path.join(parent_dir, self.results_dir, f'context_{context_language}_{needle_language}_multi_needles_{self.key_word_word}/')
                # Save the context to file for retesting
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                
                print({"results_dir":results_dir})
            

                if not os.path.exists(f'{results_dir}{context_file_location}_results.json'):
                    with open(f'{results_dir}{context_file_location}_results.json', 'w') as f:
                        f.write('[]')
                
                with open(f'{results_dir}{context_file_location}_results.json', 'r+') as f:
                    data = json.load(f)
                    data.append(results)
                    f.seek(0)  # 重置文件指针到开头
                    json.dump(data, f, ensure_ascii=False, indent=4)  # 写回修改后的数据
                    
            # if self.save_contexts:
            #     results['file_name'] = context_file_location

            #     # Save the context to file for retesting
            #     if not os.path.exists('contexts'):
            #         os.makedirs('contexts')

            #     with open(f'contexts/{context_file_location}_context.txt', 'w') as f:
            #         f.write(context)
                
            # if self.save_results:
            #     # Save the context to file for retesting
            #     if not os.path.exists('results'):
            #         os.makedirs('results')

            #     # Save the result to file for retesting
            #     with open(f'results/{context_file_location}_results.json', 'w') as f:
            #         json.dump(results, f)

            if self.seconds_to_sleep_between_completions:
                await asyncio.sleep(self.seconds_to_sleep_between_completions)

    async def bound_evaluate_and_log(self, sem, *args):
            async with sem:
                await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        
        all_combinations = product(self.context_lengths, self.document_depth_percents)

        for context_length, depth_percent in all_combinations:
            task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needles: {self.needles}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())

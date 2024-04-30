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
from itertools import product

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None,
                 needle = None,
                 haystack_dir = "PaulGrahamEssays",
                 result_dir = 'results/',
                 context_dir = 'contexts/',
                 retrieval_question = None,
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 16000,
                 context_lengths_num_intervals = 35,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 35,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 multi_needles = 0,
                 length_of_needles = 1,
                 key_word = '',
                 **kwargs):
        """
        :model_to_test: The model to test. Default is None.
        :evaluator: An evaluator to evaluate the model's response. Default is None.
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        :param kwargs: Additional arguments.
        """
        if not model_to_test:
            raise ValueError("A language model must be provided to test.")
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []
        if length_of_needles > 1:
            self.multi_needles = 1
        else:
            self.multi_needles = 0
        # self.multi_needles = multi_needles
        self.length_of_needles = length_of_needles


        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            
            if document_depth_percent_interval_type == 'linear':
                self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
            elif document_depth_percent_interval_type == 'sigmoid':
                self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
            else:
                raise ValueError("document_depth_percent_interval_type must be either 'sigmoid' or 'linear' if document_depth_percents is None.")
        else:
            self.document_depth_percents = document_depth_percents
        
        self.model_to_test = model_to_test
        self.model_name = self.model_to_test.model_name
        
        self.evaluation_model = evaluator
        
        self.start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = result_dir
        self.context_dir = context_dir
        self.key_word_word = key_word
        

    def logistic(self, x, L=100, x0=50, k=.1):
        if x in [0, 100]:
            return x
        x = -k * (x - x0)
        return np.round(L * self.sigmoid(x), 3)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    def is_chinese(self,context):
        chinese_pattern = re.compile("[\u4e00-\u9fa5]+")
        return True if chinese_pattern.search(context) else False
    
    
    def language_detection(self,context):
        if self.is_chinese(context):
            return "zh"
        else:
            return "en"
    
    
    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        
        # 生成所有参数组合
        all_combinations = product(self.context_lengths, self.document_depth_percents)

        for context_length, depth_percent in all_combinations:
            task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
            tasks.append(task)
        
        # for context_length in self.context_lengths:
        #     for depth_percent in self.document_depth_percents:
        #         for context_language in ['en', 'zh']:
        #             for needle_language in ['en', 'zh']:
        #                 task = self.bound_evaluate_and_log(sem, context_length, depth_percent, context_language, needle_language)
        #                 tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    async def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if '/' in self.evaluation_model.model_name:
            parts = self.evaluation_model.model_name.split('/')
        # 获取 '/' 后面的部分，即分割后列表的第二个元素
            model_specific_part = parts[1]
        else:
            model_specific_part = self.evaluation_model.model_name
        
        
        
        # needle语言判断
        needle_language = self.language_detection(self.needle)
        
        
        context = self.read_context_files()
        # 上下文语言判断
        context_language = self.language_detection(context)
        
        
        if self.save_results:
            if self.result_exists(context_length, depth_percent, model_specific_part, context_language, needle_language):
            
                print(f"Result exists for context_length={context_length}, depth_percent={depth_percent}, model_specific_part={model_specific_part} ,context_language= {context_language}, needle_language={needle_language}")
                return 
            else :
                print(f"Result not exists for context_length={context_length}, depth_percent={depth_percent}, model_specific_part={model_specific_part} ,context_language= {context_language}, needle_language={needle_language}")
                    
        

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)

        test_start_time = time.time()

        # Go see if the model can answer the question to pull out your random fact
        response = await self.model_to_test.evaluate_model(prompt)

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Compare the reponse to the actual needle you placed
        score, reasoning = await self.evaluation_model.evaluate_response(response)
        eval_end_time = time.time()
        eval_elapsed_time = eval_end_time - test_end_time
        
        
        
        
        
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
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
            'evaluator' : model_specific_part,
            'eval_duration_seconds' : eval_elapsed_time,
            'haystack_dir' : self.haystack_dir,
            'context_language' : context_language,
            'needle_language' : needle_language,
            'multi_needles' : self.multi_needles,
            'length_of_needles' : self.length_of_needles,
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
            
        

        # context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}_{context_language}_{self.start_time_str}'
        context_file_location = f'{self.model_name.replace(".", "_")}_{context_language}_{self.start_time_str}_{model_specific_part}_{needle_language}'
        
        base_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
        print({"base_dir":base_dir})
        if self.save_contexts:
            results['file_name'] = context_file_location

            contexts_dir = os.path.join(parent_dir, self.context_dir, f'context_{context_language}_{needle_language}_needle/')
            print({"contexts_dir":contexts_dir})
            
            # Save the context to file for retesting
            if not os.path.exists(contexts_dir):
                os.makedirs(contexts_dir)

            with open(f'{contexts_dir}_context_len_{context_length}_depth_{int(depth_percent)}%_{context_file_location}.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            results_dir = os.path.join(parent_dir, self.results_dir, f'context_{context_language}_{needle_language}_needle_{self.key_word_word}/')
            # Save the context to file for retesting
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            print({"results_dir":results_dir})
               
            
            
            # Save the result to file for retesting
            # with open(f'{results_dir}{context_file_location}_results_{needle_language}.json', 'a') as f:
            #     json.dump(self.testing_results, f)
            
            if not os.path.exists(f'{results_dir}{context_file_location}_results.json'):
                with open(f'{results_dir}{context_file_location}_results.json', 'w') as f:
                    f.write('[]')
                
            with open(f'{results_dir}{context_file_location}_results.json', 'r+') as f:
                data = json.load(f)
                data.append(results)
                f.seek(0)  # 重置文件指针到开头
                json.dump(data, f, ensure_ascii=False, indent=4)  # 写回修改后的数据
                
        if self.seconds_to_sleep_between_completions:
            await asyncio.sleep(self.seconds_to_sleep_between_completions)
            
    def check_result_match(self, result, context_length, depth_percent, evaluator_name):
        """
        Helper method to check if the results match the given parameters.
        """
        context_length_met = result['context_length'] == context_length
        depth_percent_met = result['depth_percent'] == depth_percent
        version_met = result.get('version', 1) == self.results_version
        model_met = result['model'] == self.model_name
        evaluator_met = result['evaluator'] == evaluator_name
        haystack_dir_met = result['haystack_dir'] == self.haystack_dir
        length_of_needles_met = result['length_of_needles'] == self.length_of_needles

        return all([context_length_met, depth_percent_met, version_met, model_met, evaluator_met, haystack_dir_met, length_of_needles_met])


    def result_exists(self, context_length, depth_percent, evaluator_name, context_language, needle_language):
        """
        Checks to see if a result has already been evaluated or not
        """
        base_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
        
        if self.multi_needles:
            results_dir = os.path.join(parent_dir, self.results_dir, f'context_{context_language}_{needle_language}_multi_needles_{self.key_word_word}/')
        else:
            results_dir = os.path.join(parent_dir, self.results_dir, f'context_{context_language}_{needle_language}_needle_{self.key_word_word}/')
            
        
        
        # results_dir = 'results/'
        if not os.path.exists(results_dir):
            return False
            
        # for folder in os.listdir(results_dir):
        #     if folder.endswith('bak'):
        #         continue
            # folder_path  = os.path.join(results_dir,folder)
            # print({"folder_path":folder_path})
        for filename in os.listdir(results_dir):
            file_path = os.path.join(results_dir, filename)
            
            if filename.endswith('.json'):
                if self.model_name.replace('.', '_') not in filename:
                    continue
                with open(file_path ,'r') as f:
                    try:
                        results = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from {filename}")
                        continue  # Skip this file and move to the next
                    if isinstance(results, list):
                        for result in results:
                            if self.check_result_match(result, context_length, depth_percent, evaluator_name):
                                return True

                            
                    elif isinstance(results, dict):
                        if self.check_result_match(result, context_length, depth_percent, evaluator_name):
                            return True
                        
                    else:
                        print(f"Unexpected format in {filename}")
                        return False
            
            
        return False

    async def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your haystack dir files loaded into a string
        context = self.read_context_files()
        length_txt = len(context)
        length_tokens = len(self.model_to_test.encode_text_to_tokens(context))

        # Truncate the haystack dir essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)
        
        # context = self.insert_needles(context, depth_percent, context_length)

        return context
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.model_to_test.encode_text_to_tokens(self.needle)
        tokens_context = self.model_to_test.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer
        

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]
            
            punctuations = ['. ', '。', '? ', '？', '! ', '！']
            punctuation_tokens = [self.model_to_test.encode_text_to_tokens(punc.strip()) for punc in punctuations]
            flat_list = [item for sublist in punctuation_tokens for item in sublist]
            
            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            # period_tokens = self.model_to_test.encode_text_to_tokens('. 。')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in flat_list:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.model_to_test.decode_tokens(tokens_new_context)
        return new_context

    
    def insert_needles(self, context, depth_percent, context_length):
            tokens_context = self.model_to_test.encode_text_to_tokens(context)
            context_length -= self.final_context_length_buffer

            # Calculate the total length of all needles in tokens
            total_needles_length = sum(len(self.model_to_test.encode_text_to_tokens(needle)) for needle in self.needle)

            # Ensure context length accounts for needles
            if len(tokens_context) + total_needles_length > context_length:
                tokens_context = tokens_context[:context_length - total_needles_length]
            
            # To evenly distribute the needles, we calculate the intervals they need to be inserted.
            depth_percent_interval = (100 - depth_percent) / len(self.needle)
            
            # Reset the insertion percentages list for the current context
            self.insertion_percentages = []

            # Insert needles at calculated points
            for needle in self.needle:

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
    
    
    def get_context_length_in_tokens(self, context):
        return len(self.model_to_test.encode_text_to_tokens(context))
    
    def detect_encoding(self,file_path):
        import chardet
        with open(file_path, 'rb') as file:
            # 读取足够的数据来检测编码，可以读取较大的数据以提高准确性
            data = file.read(100000)
            encoding = chardet.detect(data)['encoding']
            return encoding

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory
        context_path = os.path.join(base_dir, self.haystack_dir)
        if not os.path.exists(context_path):
            raise ValueError(f"The context directory {context_path} does not exist.")
        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(os.path.join(context_path, "*.txt")):
                original_encoding = self.detect_encoding(file)
                with open(file, 'r',encoding=original_encoding) as f:
                    context += f.read()
                    length_tokens = self.get_context_length_in_tokens(context)
        return context

    def encode_and_trim(self, context, context_length):
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self):
        
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())
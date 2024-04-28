from needlehaystack import LLMNeedleHaystackTester, LLMExamTester, LLMEvaluator, LLMMultiNeedleHaystackTester
from needlehaystack.providers import qwen
from needlehaystack.evaluators import openai, qwen_eval, together_api
import tqdm
import numpy as np
from retry import retry

# needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
# retrieval_question = "What is the best thing to do in San Francisco?"
# context_length = [400]
# depth_percent = [70]


# qwen_model = qwen.Qwen(model_name="qwen1.5-7B-Chat")
# openai_evaluator = openai.OpenAIEvaluator(model_name= "gpt-3.5-turbo-0125",
#                                           true_answer = needle,
#                                           question_asked = retrieval_question)


# ht  = LLMNeedleHaystackTester(model_to_test=qwen_model,
#                               evaluator = openai_evaluator,
#                               needle = needle,
#                               haystack= "PaulGrahamEssays",
#                               retrieval_question= retrieval_question,
#                               context_lengths = context_length,
#                               document_depth_percents = depth_percent,)
# ht.start_test()

# @retry(tries=3, delay=1)
def single_needle_test(model_name, eval_model, depth_percent, context_length, retrieval_question, needle):
    qwen_model = qwen.Qwen(model_name= model_name)
    if 'gpt' in eval_model :
        evaluator = openai.OpenAIEvaluator(model_name=eval_model,
                                                  true_answer = needle,
                                                  question_asked = retrieval_question)
    elif 'qwen' in eval_model:
        evaluator = qwen_eval.QwenEvaluator(model_name = eval_model,
                                            true_answer = needle,
                                            question_asked = retrieval_question)
    else:
        evaluator = together_api.TogetherAPIEvaluator(model_name = eval_model,
                                                      true_answer = needle,
                                                      question_asked = retrieval_question)
        
    ht  = LLMNeedleHaystackTester(model_to_test=qwen_model,
                                  evaluator = evaluator,
                                  needle = needle,
                                  haystack_dir= "PaulGrahamEssays",
                                  retrieval_question= retrieval_question,
                                  context_lengths = context_length,
                                  document_depth_percents = depth_percent,
                                  num_concurrent_requests = 20
                                  )
    # ht = LLMExamTester(model_to_test=qwen_model,
    #                       evaluator = openai_evaluator,
    #                       question = retrieval_question,
    #                       question_type = "exam",
    #                       question_path = "Exam/",
    #                       exam_results_dir = "",
    #                       exam_set = "exam",
    #                       frac =1,
    #                       num_concurrent_requests = 1
    #                       )
    print(66666)
    ht.start_test()


# 第一轮测试
# needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
# retrieval_question = "What is the best thing to do in San Francisco?"
# test_qwen([10,20,30,40,50,60,70,80,90], [500,1000,1500,2000,2500,3000,3500,4000,4500,5000], retrieval_question, needle) #在3500长度CUDA out of memory

# 第一轮补充测试
# needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n" 
# retrieval_question = "What is the best thing to do in San Francisco?"
# # test_qwen([10,20], [500,1000], retrieval_question, needle)
# test_qwen([10,30,50,70,90], [7500,8000,8500], retrieval_question, needle)  #单卡 CUDA out of memory

# 第二轮测试中文测试，语料为英文PaulGrahamEssays
# needle = "\n在旧金山最好的事情是在一个阳光明媚的日子里吃三明治并坐在多洛雷斯公园。\n"
# retrieval_question = "在旧金山最好的事情是什么？"
# haystack = "PaulGrahamEssays"
# test_qwen([30,50,70,90], [9500], retrieval_question, needle)
# test_qwen([10,30,50,70,90], [4000,4500,5000,5500,6000,6500,7000], retrieval_question, needle)


# # 第三轮测试中文测试，语料为中文设备手册
# needle = "\n在旧金山最好的事情是在一个阳光明媚的日子里吃三明治并坐在多洛雷斯公园。\n"
# retrieval_question = "在旧金山最好的事情是什么？"
# haystack = "DeviceManuals"

# 测试项设计：
# 1. 不同语料，中英文
# 2. 不同模型大小qwen7B qwen14B
# 3. 不同gpu数量,单卡和多卡
# 4. GPU和CPU推理效果对比

# for model_name in ["qwen1.5-32B-Chat-AWQ",'qwen1.5-14B-Chat',]:
# model_name = "qwen1.5-MoE-A2.7B-Chat"
# model_name = "qwen1.5-7B-Chat"
# model_name = "qwen1.5-32B-Chat-AWQ"
# model_name = 'qwen1.5-14B-Chat'
# eval_model = 'qwen1.5-14B-Chat'
# eval_model = 'gpt-3.5-turbo-0125'
# eval_model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# eval_model = 'meta-llama/Llama-3-70b-chat-hf'
eval_model = 'Qwen/Qwen1.5-72B-Chat'

needle = "\n在旧金山最好的事情是在一个阳光明媚的日子里吃三明治并坐在多洛雷斯公园。\n"
retrieval_question = "在旧金山最好的事情是什么？"


# needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
# retrieval_question = "What is the best thing to do in San Francisco?"

context_length = np.arange(500, 10000, 500).tolist()
depth_percent = np.arange(10, 100, 10).tolist()

single_needle_test(model_name, eval_model, depth_percent, context_length, retrieval_question, needle)

# for model_name in ["qwen1.5-7B-Chat", "qwen1.5-14B-Chat", "qwen1.5-MoE-A2.7B-Chat",  "qwen1.5-32B-Chat-AWQ"]:
    
#     try:
#         test_qwen(model_name, eval_model, depth_percent, context_length, retrieval_question, needle)
#     except Exception as e:
#         print(f"Error testing model {model_name}: {str(e)}") 








from needlehaystack import LLMNeedleHaystackTester, LLMExamTester, LLMEvaluator, LLMMultiNeedleHaystackTester
from needlehaystack.providers import qwen
from needlehaystack.evaluators import openai, qwen_eval, together_api
import tqdm
import numpy as np
from retry import retry


def multi_needle_test(model_name, eval_model, depth_percent, context_length,retrieval_question, needles, true_answers):
    
    
    qwen_model = qwen.Qwen(model_name= model_name)
    if 'gpt' in eval_model :
        evaluator = openai.OpenAIEvaluator(model_name=eval_model,
                                                  true_answer = true_answers,
                                                  question_asked = retrieval_question)
    elif 'qwen' in eval_model:
        evaluator = qwen_eval.QwenEvaluator(model_name = eval_model,
                                            true_answer = true_answers,
                                            question_asked = retrieval_question)
    else:
        evaluator = together_api.TogetherAPIEvaluator(model_name = eval_model,
                                                      true_answer = true_answers,
                                                      question_asked = retrieval_question)
        
    ht  = LLMMultiNeedleHaystackTester(needles = needles,
                                       model_to_test=qwen_model,
                                       evaluator = evaluator,
                                  
                                    haystack_dir= "PaulGrahamEssays",
                                    # haystack_dir= "Test",
                                    retrieval_question= retrieval_question,
                                    context_lengths = context_length,
                                    document_depth_percents = depth_percent,
                                    num_concurrent_requests = 20
                                    )
    
    ht.start_test()


# model_name = "qwen1.5-7B-Chat"
model_name = "qwen1.5-32B-Chat-AWQ"
# model_name = 'qwen1.5-14B-Chat'
# eval_model = "gpt-3.5-turbo-0125"
# eval_model = "qwen1.5-14B-Chat"
# eval_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# eval_model = "meta-llama/Llama-3-70b-chat-hf"
eval_model = "Qwen/Qwen1.5-72B-Chat"


context_length = np.arange(500, 10000, 500).tolist()
depth_percent = np.arange(10, 100, 10).tolist()
# context_length = [400]
# depth_percent = [70]                  
# retrieval_question = "What are the 3 most delicious pizza toppings?"
# true_answers = "The three most delicious pizza toppings are figs, prosciutto, and goat cheese."
# needles = ["Figs are one of the three most delicious pizza toppings.", 
#            "Prosciutto is one of the three most delicious pizza toppings.", 
#            "Goat cheese is one of the three most delicious pizza toppings."]

# multi_needle_test(model_name, eval_model, depth_percent, context_length, retrieval_question, needles, true_answers)

# for i in range(3):
# for model_name in ["qwen1.5-14B-Chat",   "qwen1.5-32B-Chat-AWQ"]:
    
# try:
#     multi_needle_test(model_name, eval_model, depth_percent, context_length, retrieval_question, needles, true_answers)
# except Exception as e:
#     print(f"Error testing model {model_name}: {str(e)}")
    
retrieval_question = "三种最美味的披萨配料是什么？"
needles = ["无花果是三种最美味的披萨配料之一。",
           "意大利熏火腿是三种最美味的披萨配料之一。",
           "羊奶酪是三种最美味的披萨配料之一。"]
true_answers = "三种最美味的披萨配料是无花果、意大利熏火腿和羊奶酪。"

multi_needle_test(model_name, eval_model, depth_percent, context_length, retrieval_question, needles, true_answers)

from needlehaystack import LLMNeedleHaystackTester, LLMExamTester, LLMEvaluator, LLMMultiNeedleHaystackTester
from needlehaystack.providers import qwen
from needlehaystack.evaluators import openai, qwen_eval, together_api
import tqdm
import numpy as np
from retry import retry

model_name = "qwen1.5-7B-Chat"

exam_path = '/home/dff652/benchmarks/LLM_Test/needlehaystack/Exam/test_0.4_en.xlsx'
exam_path = '/home/dff652/benchmarks/LLM_Test/needlehaystack/Exam/test_0.4_zh.xlsx'

qwen_model = qwen.Qwen(model_name = model_name)
ht = LLMExamTester(model_to_test=qwen_model,
                        # evaluator = openai_evaluator,
                        #   question = retrieval_question,
                        question_type = "exam",
                        question_path = exam_path,
                        exam_results_dir = "",
                        exam_set = "exam",
                        num_concurrent_requests = 50,
                        frac = 1,
                        results_version = 0.4
                        )
print(66666)
ht.start_test()

# openai_evaluator = openai.OpenAIEvaluator(model_name="gpt-3.5-turbo-0125",
#                                             true_answer = 'needle',
#                                             question_asked = 'retrieval_question'
#                                             )

qwen_evaluator = qwen_eval.QwenEvaluator(model_name=model_name)

exam_path = '/home/dff652/benchmarks/test_res/test_eng/qwen1_5-7B-Chat_question_type_exam_20240408_141046_50_human_eval.csv'

he = LLMEvaluator(evaluator = qwen_evaluator,
                  read_results_path = exam_path,
                  
                  num_concurrent_requests = 20,
                  frac = 1
                  )

print(777777)
he.start_eval()
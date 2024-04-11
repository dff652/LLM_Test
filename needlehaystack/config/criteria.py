CRITERIA_NEEDLEHAYSTACK = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}


# CRITERIA_EXAM = {
#     "evaluation_criteria": """
#     Score 0 : The chosen option is completely incorrect.
#     Score 1 : The chosen option is correct, but the explanation directly contradicts the option.
#     Score 2 : The description of key elements contains significant errors or is severely inconsistent with the question description.
#     Score 3: The description of key elements is basically accurate, but there are minor errors or omissions.
#     Score 4: The description of all key elements is completely accurate and fully consistent with the question description.
    
#     Completeness (1-4 points):
#     Score 1: The answer omits two or more key elements (device category, data phenomenon, handling measures).
#     Score 2: The answer omits one key element.
#     Score 3: The answer mentions all key elements but some parts are incomplete or not detailed enough.
#     Score 4: The answer completely mentions all key elements with detailed descriptions.
    
#     Accuracy (1-3 points):
#     Score 1: The description of key elements contains significant errors or is severely inconsistent with the question description.
#     Score 2: The description of key elements is basically accurate, but there are minor errors or omissions.
#     Score 3: The description of all key elements is completely accurate and fully consistent with the question description.
    
#     Clarity of Expression (1-3 points):
#     Score 1: The answer is expressed in a chaotic manner, the logic is unclear, and it is difficult to understand.
#     Score 2: The answer is basically clear in expression, but there are parts that are not clearly articulated or the language is not concise.
#     Score 3: The answer is expressed clearly, with fluent logic and concise language.

#     Total Evaluation Score:
#     - The total score is the sum of the scores for Completeness, Accuracy, and Clarity of Expression, 
#     adjusted to fit within a 2-10 range for correct and non-contradictory answers. 
#     For example, if the raw score sum is 10 (4 for Completeness, 3 for Accuracy, 3 for Clarity), 
#     the final score may need adjustment to fit within the 2-10 scoring range based on a predetermined scaling method.

#     Only respond with a numerical score.
#     """
# }

# CRITERIA_EXAM = {
#     "evaluation_criteria": """
#         Score 10: Correct-Concordant, The answer is not only correct but also consistent with the expected or common understanding, representing the ideal response.     
#         Score 5: Correct-Discordant, The answer is correct but does not fully align with the expected or common understanding, indicating a correct but potentially less common or less expected approach.    
#         Score 3: Incorrect-Concordant, The answer is incorrect, but the manner of the mistake is consistent with a common misunderstanding, showing a widespread but incorrect understanding.   
#         Score 1: Incorrect-Discordant, The answer is both incorrect and inconsistent with common misunderstandings, indicating a unique but incorrect interpretation.
#         respond with a numerical score and reasoning 
#     """
# }

# CRITERIA_EXAM = {
#     "evaluation_criteria": """
#         score options: 0, 0.5, 1, Dont out of options
#         Score 1: Correct-Concordant, The answer is not only correct but also consistent with the expected or common understanding, representing the ideal response.     
#         Score 0.5: Correct-Discordant, The answer is correct but does not fully align with the expected or common understanding, indicating a correct but potentially less common or less expected approach.    
#         Score 0: Incorrect-Concordant, The answer is incorrect, but the manner of the mistake is consistent with a common misunderstanding, showing a widespread but incorrect understanding.   
#         Score 0: Incorrect-Discordant, The answer is both incorrect and inconsistent with common misunderstandings, indicating a unique but incorrect interpretation.
#         respond with a numerical score and reasoning 
#     """
# }

CRITERIA_EXAM = {
    "evaluation_criteria": """
        Score 1: 只要回答中的选项正确就给1分，不用管其他内容 .     
        Score 0: 只要回答中的选项错误就给0分，不用管其他内容.   
        respond with a numerical score and reasoning 
    """
}

# CRITERIA_EXAM = {
#     "evaluation_criteria": """
#         Please provide your response in JSON format, using double quotes for keys and string values. Avoid using unescaped double quotes within strings. Each response should be a dictionary with 'score' and 'reasoning' as keys. See the example below for the correct format.
        
#         Example:
#         {
#             "score": 10,
#             "reasoning": "The answer is not only correct but also consistent with the expected or common understanding, representing the ideal response."
#         }
        
#         Evaluation Criteria:
#         Score 10: Correct-Concordant - The answer is not only correct but also consistent with the expected or common understanding, representing the ideal response.     
#         Score 5: Correct-Discordant - The answer is correct but does not fully align with the expected or common understanding, indicating a correct but potentially less common or less expected approach.    
#         Score 3: Incorrect-Concordant - The answer is incorrect, but the manner of the mistake is consistent with a common misunderstanding, showing a widespread but incorrect understanding.   
#         Score 1: Incorrect-Discordant - The answer is both incorrect and inconsistent with common misunderstandings, indicating a unique but incorrect interpretation.
        
#         Please respond with a numerical score and reasoning in dict form. Keys must be enclosed in double quotes as per JSON format requirements.
#     """
# }


# CRITERIA_EXAM = {
#     "evaluation_criteria": """
#         Score 10: Perfectly Correct-Concordant - The response is not only accurate but also perfectly aligns with the expected or common understanding, demonstrating an ideal and exemplary understanding of the subject.

#         Score 8: Mostly Correct-Concordant - The response is correct and mostly aligns with the common understanding, though it may include minor inaccuracies or less common interpretations that do not significantly detract from its overall quality.

#         Score 5: Correct-Discordant - While accurate, the response deviates from the expected or common understanding, suggesting a correct but unconventional approach or interpretation.

#         Score 4: Partially Incorrect-Concordant - The response contains significant inaccuracies, yet these inaccuracies reflect common misunderstandings or misconceptions, indicating a flawed but understandable approach.

#         Score 2: Incorrect-Concordant - The response is incorrect, and the nature of the error aligns with widespread but incorrect understandings, revealing a common but incorrect perspective.

#         Score 1: Completely Incorrect-Discordant - The response is both incorrect and uniquely misguided, showing an unusual and incorrect interpretation far removed from common understanding or misconceptions.

#         Rating must contain a doubel bracket string!
#         Please respond with a numerical score based on the above criteria, followed by comments elaborating on the reasoning behind the chosen score.
#     """
# }


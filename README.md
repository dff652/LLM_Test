#  Local LLMs Testing 

在 [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) 测试方法中加入了本地模型的测试，目前支持qwen系列(7b,14b,32b)。
评估模型可以使用本地模型或者在线api

# 目前支持的测试方法：
## 1. 考题测试
    主要测试模型知识掌握能力，支持考题导入，可进行多项测试打分，如：代码生成、头脑风暴、数学等
## 2. “海底捞针”法（NeedleInAHaystack）
    主要测试模型长文本推理能力

# Rodamap
- [x] 本地评估模型 ✓
- [x] 本地测试模型 ✓
- [ ] CPU推理测试 ✗
- [x] 性能测试 ✓


# Thanks
This project includes code derived from [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack), which is licensed under the MIT License. 

Original project authored by [gkamradt](https://github.com/gkamradt).

The original project's license file ([LICENSE](https://github.com/dff652/LLM_Test/blob/main/LICENSE.txt)) can be found in this repository.


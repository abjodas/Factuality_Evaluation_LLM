# Factuality Evaluation Using LLMs

The main goal of this project is to understand the ability of Large Language Models to judge whether a summary is consistent with a document or not. For a summary to be consistent, there must not be any extra information present in it which was not present in the document.

# Instructions

To run the project, simply clone or download this repository and then install the requirements using - 

`pip install -r requirements.txt`

You need to make a .env file where you specify your LLM api key. Make sure you name it gpt_api_key for **openai api key**, **dp_api_key** for deepseek api key, **together_api_key** for together api key and **qwen_api_key** for qwen api key.

You can specify your desired LLM provider using the argument **--llm_provider**(gpt, dp, llama, lg, qwen) and the model name using the argument **--model_name**(eg:-gpt-4.1-mini, deepseek-chat etc.).

---
You can then choose to run either the consistency evaluation task or the ranking task by passing "consistency" in the **--task** argument.
1. **Consistency** - For the consistency evaluation task, you have the option to select *cogensumm, factcc, polytope, xsumfaith* and *summeval* datasets. You can also select the split you want to choose for this task(*test* or *val*). The jsonl files required for the evaluation are given in the data folder and you do not need to download any extra files. 
2. **Ranking** - For the ranking evaluation task, you can either specify *frank* in the argument **--dataset_name** or *ranking* in the argument **--task**, specifying both will not alter this behaviour.

# Output
1. **Consistency** - You will be able to see the predictions while it is running and the final accuracy score and the balanced accuracy score once it finishes the one and only epoch.
2. **Ranking** - You will be able to see the predicted score while it is running and the Pearson Correlation Score (ρ) and Spearman Correlation Score (r) once it has finished running.

# Website
A website has been designed based on this project and you can access it [here](https://consistency-checker.onrender.com/).

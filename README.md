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

# Task Combinations for Factuality Evaluation

## Available Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `dataset_name` | `cogensumm`, `factcc`, `polytope`, `summeval`, `xsumfaith`, `frank`, `fib` | `cogensumm` | Dataset to evaluate |
| `llm_provider` | `qwen`, `gpt`, `dp`, `lg`, `llama` | `dp` | LLM provider |
| `trad_method` | `summac`, `bartscore`, `ner_consistency` | `""` | Traditional evaluation method |
| `model_name` | Any model name | `deepseek-chat` | Specific model name |
| `task` | `consistency`, `ranking`, `bartscore`, `ner_consistency`, `correlation`, `correlation_llm` | `consistency` | Task to perform |
| `split` | `train`, `val`, `test` | `val` | Dataset split |
| `type` | `COT`, `no_COT` | `COT` | Chain-of-thought or direct prompting |

## Valid Task Combinations

| Task | Dataset(s) | Required Args | Optional Args | Description |
|------|------------|---------------|---------------|-------------|
| **consistency** | `cogensumm`, `factcc`, `polytope`, `summeval`, `xsumfaith` | `--dataset_name`, `--split` | `--llm_provider`, `--model_name`, `--type` | Evaluate factual consistency using LLMs |
| **ranking** | `frank` | `--dataset_name frank` | `--llm_provider`, `--model_name` | Rank summaries by factuality (FRANK dataset) |
| **ranking** | `fib` | `--dataset_name fib` | `--llm_provider`, `--model_name`, `--type` | Binary ranking consistency (FIB dataset) |
| **ranking** | `fib` + SummaC | `--dataset_name fib`, `--trad_method summac` | `--model_name`, `--llm_provider` | Binary ranking using SummaC method |
| **bartscore** | Fixed dataset | None required | None | Evaluate using BARTScore method |
| **ner_consistency** | `factcc`, `polytope` | `--dataset_name` | `--split` | Named Entity Recognition consistency |
| **correlation** | Fixed dataset | None required | None | Correlation analysis with multiple metrics |
| **correlation_llm** | Fixed dataset | None required | `--model_name`, `--llm_provider`, `--type` | LLM-based correlation analysis |

## Example Commands

### Consistency Evaluation
```bash
# Basic consistency evaluation
python main.py --task consistency --dataset_name factcc --split val --llm_provider dp --model_name deepseek-chat

# With chain-of-thought prompting
python main.py --task consistency --dataset_name polytope --type COT --llm_provider gpt --model_name gpt-4

# Without chain-of-thought
python main.py --task consistency --dataset_name cogensumm --type no_COT --llm_provider qwen
```

### Ranking Tasks
```bash
# FRANK dataset ranking
python main.py --task ranking --dataset_name frank --llm_provider dp --model_name deepseek-chat

# FIB dataset binary ranking
python main.py --task ranking --dataset_name fib --llm_provider gpt --model_name gpt-4 --type COT

# FIB dataset with SummaC
python main.py --task ranking --dataset_name fib --trad_method summac
```

### Traditional Methods
```bash
# NER consistency evaluation
python main.py --task ner_consistency --dataset_name factcc --split test
python main.py --task ner_consistency --dataset_name polytope --split val

# BARTScore evaluation
python main.py --task bartscore

# Correlation analysis
python main.py --task correlation
python main.py --task correlation_llm --llm_provider dp --model_name deepseek-chat --type COT
```

## Dataset-Task Compatibility Matrix

| Dataset | consistency | ranking | bartscore | ner_consistency | correlation | correlation_llm |
|---------|:-----------:|:-------:|:---------:|:---------------:|:-----------:|:---------------:|
| `cogensumm` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `factcc` | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `polytope` | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `summeval` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `xsumfaith` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `frank` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `fib` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Fixed datasets | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |

## Notes

- **Fixed datasets**: Some tasks use hardcoded dataset paths (`human_annotations.aligned.paired.jsonl`, `model_annotations.aligned.paired.jsonl`)
- **LLM Providers**: `dp` = DeepSeek, `gpt` = OpenAI GPT, `qwen` = Qwen, `lg` = Together API, `llama` = Llama via Together
- **Chain-of-Thought**: `COT` enables reasoning steps, `no_COT` uses direct prompting
- **Traditional Methods**: `summac`, `bartscore`, `ner_consistency` don't require LLM providers
- **Output Files**: Results are saved to CSV files with names based on the task and method used

# Factuality Evaluation Using LLMs

The main goal of this project is to understand the ability of Large Language Models to judge whether a summary is consistent with a document or not. For a summary to be consistent, there must not be any extra information present in it which was not present in the document.

## Quick Start

### Installation

Clone or download this repository and install the requirements:

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your LLM API keys:

```env
gpt_api_key=your_openai_api_key
dp_api_key=your_deepseek_api_key
together_api_key=your_together_api_key
qwen_api_key=your_qwen_api_key
```

### Basic Usage

Choose your LLM provider with `--llm_provider` and model with `--model_name`:

```bash
python main.py --task consistency --dataset_name factcc --llm_provider dp --model_name deepseek-chat
```

## Available Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--dataset_name` | `cogensumm`, `factcc`, `polytope`, `summeval`, `xsumfaith`, `frank`, `fib` | `cogensumm` | Dataset to evaluate |
| `--llm_provider` | `qwen`, `gpt`, `dp`, `lg`, `llama` | `dp` | LLM provider |
| `--trad_method` | `summac`, `bartscore`, `ner_consistency` | `""` | Traditional evaluation method |
| `--model_name` | Any model name | `deepseek-chat` | Specific model name |
| `--task` | `consistency`, `ranking`, `bartscore`, `ner_consistency`, `correlation`, `correlation_llm` | `consistency` | Task to perform |
| `--split` | `train`, `val`, `test` | `val` | Dataset split |
| `--type` | `COT`, `no_COT` | `COT` | Chain-of-thought or direct prompting |

### LLM Provider Codes

- `dp` = DeepSeek
- `gpt` = OpenAI GPT  
- `qwen` = Qwen
- `lg` = Together API
- `llama` = Llama via Together API

## Task Types and Execution Paths

### 1. Consistency Evaluation

Evaluates whether summaries are factually consistent with source documents.

#### Supported Datasets
- **cogensumm**: Uses `consistency_evaluator_doctype()` function
- **factcc**: Uses `consistency_evaluator_factcc()` function  
- **polytope**, **summeval**, **xsumfaith**: Use standard consistency evaluation

#### Example Commands
```bash
# CoGenSumm dataset
python main.py --task consistency --dataset_name cogensumm --split val

# FactCC dataset
python main.py --task consistency --dataset_name factcc --split test --llm_provider gpt --model_name gpt-4

# With/without chain-of-thought
python main.py --task consistency --dataset_name polytope --type COT
python main.py --task consistency --dataset_name summeval --type no_COT
```

### 2. Ranking Tasks

Ranks summaries by factuality or evaluates binary ranking consistency.

#### FRANK Dataset Ranking
```bash
python main.py --task ranking --dataset_name frank --llm_provider dp --model_name deepseek-chat
```

#### FIB Dataset Binary Ranking
```bash
# Standard LLM-based ranking
python main.py --task ranking --dataset_name fib --llm_provider gpt --model_name gpt-4

# Using SummaC traditional method
python main.py --task ranking --dataset_name fib --trad_method summac
```

### 3. Traditional Methods

#### Named Entity Recognition Consistency
```bash
# FactCC dataset
python main.py --task ner_consistency --dataset_name factcc --split test

# Polytope dataset  
python main.py --task ner_consistency --dataset_name polytope --split val
```

#### BARTScore Evaluation
```bash
python main.py --task bartscore
```

### 4. Correlation Analysis

#### Traditional Metrics Correlation
```bash
python main.py --task correlation
```

#### LLM-based Correlation
```bash
python main.py --task correlation_llm --llm_provider dp --model_name deepseek-chat --type COT
```

## Implementation Details

### File Paths and Data Loading

The code uses these specific data paths:
- CoGenSumm: `data/{dataset_name}_{split}.jsonl`
- FactCC: Loaded via HuggingFace `mtc/factcc_annotated_eval_data`
- FRANK: `data/benchmark_data.json`
- BARTScore: `data/human_annotations.aligned.paired.jsonl`
- Correlation: `data/model_annotations.aligned.paired.jsonl`
- Polytope: `data/polytope_{split}.jsonl`
- FIB: Loaded via HuggingFace `r-three/fib` (600 samples, seed=32)

### Dataset-Task Compatibility Matrix

| Dataset | consistency | ranking | bartscore | ner_consistency | correlation | correlation_llm |
|---------|:-----------:|:-------:|:---------:|:---------------:|:-----------:|:---------------:|
| `cogensumm` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `factcc` | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `polytope` | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `summeval` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `xsumfaith` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `frank` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `fib` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `tldr` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Fixed datasets* | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |


## Output and Results

### Consistency Evaluation
- **Live feedback**: Predictions displayed during execution
- **Final metrics**: Accuracy score and balanced accuracy score after completion

### Ranking Tasks  
- **Live feedback**: Predicted scores shown during execution
- **Final metrics**: Pearson Correlation Score (ρ) and Spearman Correlation Score (r)
- **Output files**: Results saved to CSV files (e.g., `fib_ranking_results.csv`)

### NER Consistency
- **Best threshold**: Optimal threshold value for classification
- **Results DataFrame**: Detailed performance metrics

## Advanced Examples

### Complete Evaluation Pipeline
```bash
# Evaluate multiple datasets with different models
python main.py --task consistency --dataset_name factcc --llm_provider dp --model_name deepseek-chat --type COT
python main.py --task consistency --dataset_name cogensumm --llm_provider gpt --model_name gpt-4 --type no_COT
python main.py --task ranking --dataset_name fib --llm_provider qwen --model_name qwen-max

# Traditional method comparison
python main.py --task ner_consistency --dataset_name factcc --split test
python main.py --task ranking --dataset_name fib --trad_method summac
```

### Correlation Studies
```bash
# Compare different evaluation approaches
python main.py --task correlation
python main.py --task correlation_llm --type COT --llm_provider dp --model_name deepseek-chat
```

## Web Interface

A web-based interface for this project is available at: [https://consistency-checker.onrender.com/](https://consistency-checker.onrender.com/)

## Notes

- **Chain-of-Thought**: `COT` enables step-by-step reasoning, `no_COT` uses direct prompting
- **Data Requirements**: Most datasets are included in the `

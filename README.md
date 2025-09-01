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

| Argument         | Options                                                                                    | Default         | Description                          |
| ---------------- | ------------------------------------------------------------------------------------------ | --------------- | ------------------------------------ |
| `--dataset_name` | `cogensumm`, `factcc`, `polytope`, `summeval`, `xsumfaith`, `frank`, `fib`                 | `cogensumm`     | Dataset to evaluate                  |
| `--llm_provider` | `qwen`, `gpt`, `dp`, `lg`, `llama`                                                         | `dp`            | LLM provider                         |
| `--trad_method`  | `summac`, `bartscore`, `ner_consistency`                                                   | `""`            | Traditional evaluation method        |
| `--model_name`   | Any model name                                                                             | `deepseek-chat` | Specific model name                  |
| `--task`         | `consistency`, `ranking`, `bartscore`, `ner_consistency`, `correlation`, `correlation_llm` | `consistency`   | Task to perform                      |
| `--split`        | `train`, `val`, `test`                                                                     | `val`           | Dataset split                        |
| `--type`         | `COT`, `no_COT`                                                                            | `COT`           | Chain-of-thought or direct prompting |

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

| Dataset          | consistency | ranking | bartscore | ner_consistency | correlation | correlation_llm |
| ---------------- | :---------: | :-----: | :-------: | :-------------: | :---------: | :-------------: |
| `cogensumm`      |     ✅      |   ❌    |    ❌     |       ❌        |     ❌      |       ❌        |
| `factcc`         |     ✅      |   ❌    |    ❌     |       ✅        |     ❌      |       ❌        |
| `polytope`       |     ❌      |   ❌    |    ❌     |       ✅        |     ❌      |       ❌        |
| `summeval`       |     ❌      |   ❌    |    ❌     |       ❌        |     ❌      |       ❌        |
| `xsumfaith`      |     ❌      |   ❌    |    ❌     |       ❌        |     ❌      |       ❌        |
| `frank`          |     ❌      |   ✅    |    ❌     |       ❌        |     ❌      |       ❌        |
| `fib`            |     ❌      |   ✅    |    ❌     |       ❌        |     ❌      |       ❌        |
| `tldr`           |     ❌      |   ✅    |    ❌     |       ❌        |     ❌      |       ❌        |
| Fixed datasets\* |     ❌      |   ❌    |    ✅     |       ❌        |     ✅      |       ✅        |

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

## GRPO Training for Consistency Models

### Overview

This project includes Group Relative Policy Optimization (GRPO) training capabilities for improving consistency evaluation models. GRPO uses multiple reward functions to train models to provide better reasoning and more accurate consistency judgments.

### Prerequisites

Before running GRPO training, ensure you have the additional dependencies:

```bash
pip install -r requirements_grpo.txt
```

### Data Preparation

The GRPO training script expects two specific CSV files in the `data/` folder:

#### 1. Initial Processing Data: `data/cot_deepseek.csv`

Required columns:

- `problems` - The consistency evaluation prompts
- `solutions` - Model-generated solutions for label extraction

#### 2. Main Training Data: `data/processed_cot_data.csv`

Required columns:

- `problem` - The consistency evaluation prompt
- `answer` - Ground truth label ("consistent" or "inconsistent")
- `reasoning` - Chain-of-thought reasoning text for SFT

Example data format for `processed_cot_data.csv`:

```csv
problem,answer,reasoning
"Decide if the following summary is consistent with the article...",consistent,"Let me analyze this step by step. First, I'll check each claim in the summary..."
```

**Data Location**: All training data files are located in the `data/` folder as referenced in the script.

### GRPO Training Process

The training consists of two phases:

#### 1. Supervised Fine-Tuning (SFT)

First, the model is fine-tuned on formatted examples:

```python
# This runs automatically before GRPO
trainer_sft = SFTTrainer(
    model=model,
    train_dataset=dataset_sft,
    args=SFTConfig(...)
)
trainer_sft.train()
```

#### 2. GRPO Training

Then GRPO training optimizes the model using multiple reward functions:

```python
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[...],  # Multiple reward functions
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### Reward Functions

The GRPO setup includes four specialized reward functions:

1. **Format Consistency** (`check_consistency_format`)

   - Rewards proper reasoning → answer structure
   - Bonus for multi-step reasoning
   - Penalty for invalid formats

2. **Answer Accuracy** (`check_consistency_answer`)

   - +10 points for correct answers
   - -5 points for no answer found
   - -3 points for wrong answers

3. **Reasoning Quality** (`check_reasoning_quality`)

   - Rewards key concepts (summary, article, evidence)
   - Bonus for structured thinking
   - Penalty for very short responses

4. **Evidence Usage** (`check_evidence_usage`)
   - Rewards references to source material
   - Bonus for comparison language and quotations
   - Encourages grounded analysis

### Configuration Options

Key training parameters you can adjust:

```python
training_args = GRPOConfig(
    learning_rate=5e-6,           # Lower LR for stable training
    per_device_train_batch_size=1, # Adjust based on GPU memory
    gradient_accumulation_steps=4,  # Effective batch size = 4
    num_generations=8,             # Completions per prompt
    max_steps=3000,               # Total training steps
    temperature=1.0,              # Sampling temperature
    max_prompt_length=1024,       # Adjust based on your data
    max_completion_length=1024,   # Max response length
)
```

### Running GRPO Training

1. **Ensure data files are in place**:

   - `data/cot_deepseek.csv` (for label extraction)
   - `data/processed_cot_data.csv` (main training data)

2. **Run the complete training script**:

```bash
python grpo_training.py
```

The script automatically handles:

- Model and tokenizer initialization (Qwen3-4B-Base with LoRA)
- SFT phase training (2 epochs on 100 samples)
- GRPO phase training (3000 steps with 4 reward functions)
- Model saving to `outputs/consistency_grpo/final_model_last`

### Expected Output

During training, you'll see:

- **SFT Phase**: Loss decreasing as model learns the format
- **GRPO Phase**: Reward scores for each function
- **Periodic saves** at `outputs/consistency_grpo/`
- **Final model** saved as `outputs/consistency_grpo/final_model_last`

### Model Testing

After training, test your model with:

```python
def test_model_output(prompt_text):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=256)

    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

# Test consistency evaluation
test_prompt = """Decide if the following summary is consistent with the article.
Note that consistency means all information in the summary is supported by the article.
<Article>...</Article>
<Summary>...</Summary>"""

print("Model output:", test_model_output(test_prompt))
```

## Comprehensive Model Evaluation

### GRPO Test Script

After GRPO training, use `grpo_test.py` for comprehensive evaluation with statistical analysis:

```bash
python grpo_test.py
```

Our trained model can be found on [https://drive.google.com/drive/folders/1FVSjAYQQ3R4aKor2WSeBj3vBvYYEEESD?usp=drive_link](https://drive.google.com/drive/folders/1FVSjAYQQ3R4aKor2WSeBj3vBvYYEEESD?usp=drive_link)
Make a directory named outputs in the same directory and place the downloaded model there

### Test Script Features

The test script provides:

1. **Multi-seed evaluation** - Tests model consistency across different random seeds
2. **Automatic text truncation** - Handles long articles/summaries that exceed context length
3. **Statistical analysis** - Calculates mean accuracy, standard deviation, and confidence intervals
4. **Robust error handling** - Skips problematic samples and reports success rates
5. **Detailed logging** - Progress tracking with token counts and intermediate accuracy

### Key Components

#### Smart Text Truncation

```python
def create_prompt_with_truncation(item, system_prompt, tokenizer, max_length=2048):
    # Allocates 80% of available tokens to article, 20% to summary
    # Automatically handles context length constraints
```

#### Multi-Experiment Runner

```python
def run_multiple_experiments(seeds, num_samples=1000, max_length=2048):
    # Runs evaluation with different random seeds for robust statistics
    # Default: 8 different seeds with 1000 samples each
```

### Expected Output

The test script provides comprehensive statistics:

```
FINAL RESULTS
============================================================
Seeds used: [3407, 42, 1337, 2023, 9999, 20, 1023, 40055]
Successful experiments: 8/8
Samples per experiment: 1000

Processing Statistics:
Total items processed: 7,856
Total items skipped: 144
Average processed per experiment: 982.0
Success rate: 98.2%

Accuracy Results:
Mean accuracy: 0.7543 ± 0.0127
Min accuracy: 0.7321
Max accuracy: 0.7689
95% confidence interval: 0.7543 ± 0.0249

Results saved to 'accuracy_results.json'
```

### Configuration Options

Customize evaluation parameters:

```python
# Test with different settings
seeds = [3407, 42, 1337, 2023, 9999]  # Custom seed list
results = run_multiple_experiments(
    seeds=seeds,
    num_samples=500,      # Reduce for faster testing
    max_length=2048       # Adjust based on your model
)
```

### Model Loading

The test script expects a trained LoRA adapter. Update the model loading line:

```python
output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("final_model_long")  # Your trained model name
)
```

### Results Analysis

The script generates:

- **Individual accuracy scores** for each seed
- **Mean and standard deviation** across experiments
- **95% confidence intervals** for robust evaluation
- **Processing statistics** showing successful vs. skipped samples
- **JSON output file** (`accuracy_results.json`) for further analysis

### Troubleshooting Testing

**Long Context Issues**: The script automatically truncates text, but you can increase `max_length` if you have sufficient GPU memory

**Memory Errors**: Reduce `num_samples` or use smaller batch processing

**Model Loading Errors**: Ensure the LoRA adapter path matches your trained model location

**Low Success Rate**: Check your data format and model compatibility

This comprehensive testing approach provides statistically robust evaluation of your GRPO-trained consistency models.

### Performance Expectations

GRPO training typically improves:

- **Response format consistency** (proper reasoning → answer structure)
- **Answer accuracy** on consistency judgments
- **Reasoning quality** with better evidence usage
- **Structured thinking** with step-by-step analysis

### Troubleshooting

**Memory Issues**: Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps`

**Training Instability**: Lower `learning_rate` (try 1e-6) or reduce `temperature`

**Poor Convergence**: Increase `max_steps` or adjust reward function weights

**Format Issues**: Ensure your CSV has the exact column names: `problem`, `answer`, `reasoning`

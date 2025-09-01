from unsloth import FastLanguageModel
import pandas as pd
import re
import torch
from datasets import Dataset
import numpy as np
import re
from vllm import SamplingParams  
from trl import GRPOConfig, GRPOTrainer


max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 256 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)
reasoning_start = "<start_working_out>" # Acts as <think>
reasoning_end   = "<end_working_out>"   # Acts as </think>
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
system_prompt
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

# Replace with out specific template:
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template
tokenizer.apply_chat_template([
    {"role" : "user", "content" : "What is 1+1?"},
    {"role" : "assistant", "content" : f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
    {"role" : "user", "content" : "What is 2+2?"},
], tokenize = False, add_generation_prompt = True)
CONSISTENT_PATTERNS = re.compile(
    r'(?:'
    r'\bAnswer:\s*\*{0,2}[Cc]onsistent\*{0,2}\b|'
    r'\bFinal\s+[Aa]nswer:\s*\*{0,2}[Cc]onsistent\*{0,2}\b|'
    r'\*{0,2}Answer\*{0,2}:\s*\*{0,2}[Cc]onsistent\*{0,2}|'
    r'\*{0,2}Final\s+answer:\s*\*{0,2}[Cc]onsistent\*{0,2}|'
    r'\*{0,2}[Cc]onsistency\*{0,2}$|'
    r'\b[Cc]onsistent$|'
    r'Answer:\s*\n\*{0,2}[Cc]onsistent\*{0,2}'
    r')',
    re.IGNORECASE | re.MULTILINE
)

def extract_answer_qwen(text: str) -> int:
    """Extract consistency prediction from model response.
    
    Args:
        text: Model response text
        
    Returns:
        1 if consistent, 0 if inconsistent
    """
    return 1 if CONSISTENT_PATTERNS.search(text) else 0
def add_labels(x):
    label = extract_answer_qwen(x['solutions'])
    return label

df = pd.read_csv('cot_deepseek.csv', usecols=['problems', 'solutions'])
df['labels'] = df.apply(add_labels, axis=1)

def format_dataset(x):
    expected_answer = x['answer'].lower()
    problem = x['problem']
    thoughts = x['reasoning']
    final_prompt = \
        reasoning_start + thoughts + reasoning_end + \
        solution_start + expected_answer + solution_end
    return [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : final_prompt},
    ]
df = pd.read_csv('processed_cot_data.csv')
df["Messages"] = df.apply(format_dataset, axis=1)

df['text'] = tokenizer.apply_chat_template(df["Messages"].values.tolist(), tokenize=False)
dataset_sft = Dataset.from_pandas(df)
dataset_sft = dataset_sft.shuffle(seed=3407).select(range(100))

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_sft,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 2, # Set this for 1 full training run.
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

trainer.train()



# === GRPO Training Setup ===

dataset = Dataset.from_pandas(df)
# === IMPORTANT: Rename columns to match GRPO expectations ===
dataset = dataset.rename_columns({
    'problem': 'prompt',
    'answer': 'answer'  # Keep answer as is
})

# === Define max sequence length ===
max_seq_length = 2048  # Adjust based on your model's context length

print("Dataset columns after renaming:", dataset.column_names)
print("Sample data:")
print("Prompt:", dataset[0]['prompt'][:100] + "...")
print("Answer:", dataset[0]['answer'])

# === Data Preprocessing ===
# Tokenize your dataset to measure lengths
tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(
        [{"role": "user", "content": x["prompt"]}], 
        add_generation_prompt=True, 
        tokenize=True
    )}
)

print("Sample tokenized prompt:")
print(tokenizer.decode(tokenized[0]["tokens"]))

# Calculate sequence lengths
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

# Find 90th percentile length to filter outliers
maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length (90th percentile) =", maximum_length)

# Filter dataset to remove very long sequences
dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

# Set length constraints
max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

print(f"Max prompt length: {max_prompt_length}")
print(f"Max completion length: {max_completion_length}")

# === Reward Functions for Consistency Task ===

def check_consistency_format(prompts, completions, answer, **kwargs):
    """
    Check if the model follows the expected reasoning + answer format.
    """
    # completions is a simple list of strings
    responses = completions
    
    # Pattern to match the expected format: reasoning followed by answer
    format_pattern = r'(.*?)\s*(?:answer|conclusion|final answer):\s*(consistent|inconsistent)'
    
    scores = []
    for response in responses:
        response_clean = response.lower().strip()
        
        # Check if response has the expected structure
        match = re.search(format_pattern, response_clean, re.DOTALL | re.IGNORECASE)
        
        if match:
            reasoning_part = match.group(1).strip()
            answer_part = match.group(2).strip()
            
            # Reward good format
            score = 3.0
            
            # Bonus for substantial reasoning (not just one sentence)
            if len(reasoning_part.split('.')) >= 3:
                score += 2.0
            
            # Bonus for structured reasoning indicators
            structure_words = ['first', 'second', 'step', 'analyze', 'examine', 'therefore']
            if any(word in reasoning_part for word in structure_words):
                score += 1.0
                
        else:
            # Check if at least the answer is present
            if any(word in response_clean for word in ['consistent', 'inconsistent']):
                score = 1.0  # Partial credit
            else:
                score = -3.0  # No valid format
        
        scores.append(score)
    
    return scores

def check_consistency_answer(prompts, completions, answer, **kwargs):
    """
    Check if the final answer matches the ground truth.
    """
    # completions is a simple list of strings
    responses = completions
    
    # Multiple patterns to extract the answer
    answer_patterns = [
        r'answer:\s*\*?\*?(consistent|inconsistent)\*?\*?',  # Handle **consistent** format
        r'conclusion:\s*\*?\*?(consistent|inconsistent)\*?\*?', 
        r'final answer:\s*\*?\*?(consistent|inconsistent)\*?\*?',
        r'therefore.*?(consistent|inconsistent)',
        r'(consistent|inconsistent)(?:\s*[.]?\s*$)',  # At the end
    ]
    
    def extract_answer(response):
        response_lower = response.lower().strip()
        
        for pattern in answer_patterns:
            match = re.search(pattern, response_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: simple word search (prioritize inconsistent as it's more specific)
        if 'inconsistent' in response_lower:
            return 'inconsistent'
        elif 'consistent' in response_lower:
            return 'consistent'
        return None
    
    scores = []
    for response, true_answer in zip(responses, answer):
        predicted = extract_answer(response)
        true_clean = true_answer.lower().strip()
        
        if predicted is None:
            score = -5.0  # No answer found
        elif predicted == true_clean:
            score = 10.0  # Correct answer
        else:
            score = -3.0  # Wrong answer
            
        scores.append(score)
    
    return scores

def check_reasoning_quality(prompts, completions, answer, **kwargs):
    """
    Evaluate the quality of reasoning provided.
    """
    # completions is a simple list of strings
    responses = completions
    
    scores = []
    for response in responses:
        response_lower = response.lower()
        score = 0
        
        # Reward mentions of key concepts
        key_concepts = ['summary', 'article', 'information', 'supported', 'evidence']
        score += sum(2.0 for concept in key_concepts if concept in response_lower)
        
        # Reward structured thinking
        structure_indicators = ['step by step', 'first', 'second', 'analyze', 'examine', 'let\'s analyze']
        score += sum(1.5 for indicator in structure_indicators if indicator in response_lower)
        
        # Reward logical connectors
        logical_words = ['therefore', 'because', 'since', 'however', 'given that']
        score += sum(1.0 for word in logical_words if word in response_lower)
        
        # Reward numbered/structured analysis (like "1.", "2.", "3.")
        numbered_points = len(re.findall(r'\d+\.', response))
        score += min(numbered_points * 0.5, 3.0)  # Up to 3 bonus points
        
        # Penalty for very short responses
        if len(response.split()) < 20:
            score -= 2.0
        
        # Cap the score
        score = min(score, 8.0)
        scores.append(score)
    
    return scores

def check_evidence_usage(prompts, completions, answer, **kwargs):
    """
    Check if the model properly references the article/summary.
    """
    # completions is a simple list of strings
    responses = completions
    
    scores = []
    for response in responses:
        response_lower = response.lower()
        score = 0
        
        # Reward explicit references to source material
        references = ['article states', 'summary claims', 'according to', 'mentioned in', 'the text', 'the article']
        score += sum(2.0 for ref in references if ref in response_lower)
        
        # Reward comparison language
        comparisons = ['matches', 'contradicts', 'supports', 'aligns with', 'differs from', 'consistent with']
        score += sum(1.5 for comp in comparisons if comp in response_lower)
        
        # Reward specific analysis terms
        analysis_terms = ['claim', 'statement', 'assertion', 'fact', 'detail']
        score += sum(1.0 for term in analysis_terms if term in response_lower)
        
        # Reward quotation usage (shows they're referencing specific text)
        quote_bonus = len(re.findall(r'"[^"]*"', response)) * 0.5
        score += min(quote_bonus, 2.0)  # Up to 2 bonus points
        
        scores.append(min(score, 6.0))
    
    return scores

# === VLLM Sampling Parameters ===
vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

# === Training Configuration ===
training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Increased for smoother training
    num_generations=8,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=3000,  # Adjust based on your dataset size
    save_steps=500,   # Save more frequently
    report_to="wandb",  # Change to "wandb" if you want logging
    output_dir="outputs/consistency_grpo",
)

# === Initialize Trainer ===
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        check_consistency_format,     # Rewards proper format
        check_consistency_answer,     # Rewards correct answers
        check_reasoning_quality,      # Rewards good reasoning
        check_evidence_usage,         # Rewards proper evidence usage
    ],
    args=training_args,
    train_dataset=dataset,
)

# === Start Training ===
print("Starting GRPO training for consistency task...")
print(f"Dataset size: {len(dataset)}")
print(f"Reward functions: {len(trainer.reward_funcs)}")

trainer.train()

# === Save the final model ===
trainer.save_model("outputs/consistency_grpo/final_model_last")
print("Training completed and model saved!")

# === Optional: Test the trained model ===
def test_model_output(prompt_text):
    """Quick test function to see model output"""
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], 
        add_generation_prompt=True, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response

# Example usage after training:
# test_prompt = "Decide if the following summary is consistent with the article..."
# print("Model output:", test_model_output(test_prompt))

print("GRPO training setup complete!")
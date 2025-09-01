from unsloth import FastLanguageModel
import torch
import re
import numpy as np
from datasets import load_dataset
from tqdm import tqdm, trange
from sklearn.metrics import balanced_accuracy_score
from vllm import SamplingParams

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower
reasoning_start = "<start_working_out>" # Acts as <think>
reasoning_end   = "<end_working_out>"   # Acts as </think>
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

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



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
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
# Replace with out specific template:
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template
answer_patterns = [
        r'answer:\s*\*?\*?(consistent|inconsistent)\*?\*?',  # Handle **consistent** format
        r'conclusion:\s*\*?\*?(consistent|inconsistent)\*?\*?', 
        r'final answer:\s*\*?\*?(consistent|inconsistent)\*?\*?',
        r'therefore.*?(consistent|inconsistent)',
        r'(consistent|inconsistent)(?:\s*[.]?\s*$)',  # At the end
    ]
    
import re
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


def truncate_text(text, max_tokens=1500):
    """Truncate text to approximately max_tokens by splitting and rejoining"""
    words = text.split()
    # Rough approximation: 1 token ≈ 0.75 words for English
    max_words = int(max_tokens * 0.75)
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + "..."
    return text

def create_prompt_with_truncation(item, system_prompt, tokenizer, max_length=2048):
    """Create a prompt that fits within the model's context length"""
    
    # Reserve space for system prompt, user prompt template, and generation
    base_prompt = f"""Decide if the following summary is consistent with the corresponding article.
Note that consistency means all information in the summary is supported by the article.
Explain your reasoning step by step first, and then answer (consistent or inconsistent) at the end:
<Article>
{{article}}
</Article>
<Summary>
{{summary}}
</Summary>
Answer:"""
    
    messages_template = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_prompt}
    ]
    
    # Calculate base tokens (without article/summary content)
    base_text = tokenizer.apply_chat_template(
        messages_template, 
        add_generation_prompt=True, 
        tokenize=False
    )
    base_tokens = len(tokenizer.encode(base_text))
    
    # Reserve tokens for generation (response)
    generation_buffer = 512
    available_tokens = max_length - base_tokens - generation_buffer
    
    # Split available tokens between article and summary (prioritize article)
    article_tokens = int(available_tokens * 0.8)  # 80% for article
    summary_tokens = available_tokens - article_tokens  # 20% for summary
    
    # Truncate content
    truncated_article = truncate_text(item['context'], article_tokens)
    truncated_summary = truncate_text(item['summary'], summary_tokens)
    
    # Create final messages
    final_prompt = base_prompt.replace("{article}", truncated_article).replace("{summary}", truncated_summary)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt}
    ]
    
    return messages

def run_single_experiment(seed, num_samples=1000, max_length=2048):
    """Run a single experiment with a given seed and return the balanced accuracy."""
    
    # Load and shuffle dataset with the given seed
    dataset = load_dataset("achandlr/FactualConsistencyScoresTextSummarization", split='train')
    dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    
    predictions = []
    true_labels = []
    skipped_items = 0
    
    with trange(len(dataset), desc=f"Seed {seed}") as t:
        for i in t:
            item = dataset[i]
            
            try:
                # Create truncated prompt
                messages = create_prompt_with_truncation(item, system_prompt, tokenizer, max_length=max_length)
                
                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                
                # Double-check the length
                token_count = len(tokenizer.encode(text))
                if token_count > max_length:
                    print(f"Warning: Item {i} still too long ({token_count} tokens), skipping...")
                    skipped_items += 1
                    continue
                
                sampling_params = SamplingParams(
                    temperature=1.0,
                    top_k=50,
                    max_tokens=512,  # Reduced max_tokens for response
                )
                
                output = model.fast_generate(
                    text,
                    sampling_params=sampling_params,
                    lora_request=model.load_lora("outputs/final_model_long"), # Download the model from https://drive.google.com/drive/folders/1FVSjAYQQ3R4aKor2WSeBj3vBvYYEEESD?usp=sharing
                )[0].outputs[0].text
                print(f"Output: {output}", flush=True)
                answer = extract_answer(output)
                dataset[i]['model_answer'] = answer
                
                true_labels.append(item['label'])
                predictions.append(1 if answer == 'consistent' else 0)
                
                if i % 10 == 0 and i > 0:
                    current_acc = balanced_accuracy_score(true_labels, predictions)
                    t.set_postfix(acc=current_acc, skipped=skipped_items, tokens=token_count)
                    
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                skipped_items += 1
                continue
    
    # Calculate final accuracy
    if len(predictions) > 0:
        final_accuracy = balanced_accuracy_score(true_labels, predictions)
        print(f"Seed {seed} - Processed: {len(predictions)}, Skipped: {skipped_items}, Accuracy: {final_accuracy:.4f}")
        return final_accuracy, len(predictions), skipped_items
    else:
        print(f"Seed {seed} - No items were successfully processed!")
        return None, 0, skipped_items

def run_multiple_experiments(seeds, num_samples=1000, max_length=2048):
    """Run multiple experiments and calculate statistics."""
    
    accuracies = []
    processed_counts = []
    skipped_counts = []
    
    print(f"Running {len(seeds)} experiments with seeds: {seeds}")
    print(f"Using {num_samples} samples per experiment")
    print(f"Max context length: {max_length} tokens\n")
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Running experiment with seed: {seed}")
        print(f"{'='*50}")
        
        result = run_single_experiment(seed, num_samples, max_length)
        accuracy, processed, skipped = result
        
        if accuracy is not None:
            accuracies.append(accuracy)
            processed_counts.append(processed)
            skipped_counts.append(skipped)
            
            print(f"Experiment {seed} completed successfully")
            
            if len(accuracies) > 1:
                current_mean = np.mean(accuracies)
                current_std = np.std(accuracies, ddof=1)  # Sample standard deviation
                print(f"Running mean: {current_mean:.4f} ± {current_std:.4f}")
        else:
            print(f"Experiment {seed} failed - no valid predictions")
    
    if len(accuracies) == 0:
        print("\nERROR: No experiments completed successfully!")
        return None
    
    # Calculate final statistics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies, ddof=1)  # Sample standard deviation
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)
    
    total_processed = sum(processed_counts)
    total_skipped = sum(skipped_counts)
    avg_processed = np.mean(processed_counts)
    avg_skipped = np.mean(skipped_counts)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Seeds used: {seeds}")
    print(f"Successful experiments: {len(accuracies)}/{len(seeds)}")
    print(f"Samples per experiment: {num_samples}")
    print(f"Max context length: {max_length} tokens")
    print(f"\nProcessing Statistics:")
    print(f"Total items processed: {total_processed}")
    print(f"Total items skipped: {total_skipped}")
    print(f"Average processed per experiment: {avg_processed:.1f}")
    print(f"Average skipped per experiment: {avg_skipped:.1f}")
    print(f"Success rate: {avg_processed/(avg_processed + avg_skipped)*100:.1f}%")
    
    print(f"\nAccuracy Results:")
    print(f"Individual accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"Standard deviation: {std_accuracy:.4f}")
    print(f"Min accuracy: {min_accuracy:.4f}")
    print(f"Max accuracy: {max_accuracy:.4f}")
    print(f"Range: {max_accuracy - min_accuracy:.4f}")
    print(f"\nConfidence interval (±1 std): {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"95% confidence interval (±1.96 std): {mean_accuracy:.4f} ± {1.96 * std_accuracy:.4f}")
    
    return {
        'accuracies': accuracies,
        'mean': mean_accuracy,
        'std': std_accuracy,
        'min': min_accuracy,
        'max': max_accuracy,
        'seeds': seeds[:len(accuracies)],  # Only successful seeds
        'processed_counts': processed_counts,
        'skipped_counts': skipped_counts,
        'total_processed': total_processed,
        'total_skipped': total_skipped
    }

# Example usage:
if __name__ == "__main__":
    # Define seeds for reproducibility
    seeds = [3407, 42, 1337, 2023, 9999, 20, 1023, 40055]  # You can modify this list
    
    # Alternative: Generate random seeds
    # np.random.seed(42)  # For reproducibility of seed generation
    # seeds = np.random.randint(0, 10000, size=5).tolist()
    
    # Run experiments
    results = run_multiple_experiments(seeds, num_samples=1000, max_length=2048)
    
    if results is not None:
        # Save results if needed
        import json
        with open('accuracy_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to 'accuracy_results.json'")
        
        # Quick summary for easy reference
        print(f"\n{'='*40}")
        print("SUMMARY")
        print(f"{'='*40}")
        print(f"Mean Accuracy: {results['mean']:.4f} ± {results['std']:.4f}")
        print(f"Processing Success Rate: {results['total_processed']/(results['total_processed'] + results['total_skipped'])*100:.1f}%")

# Alternative approach: Increase model's max_model_len if you have enough memory
"""
# If you want to handle longer sequences, you can reinitialize the model with a larger context:
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = 4096,  # Increase this
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,
)
"""
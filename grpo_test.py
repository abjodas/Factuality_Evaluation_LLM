from unsloth import FastLanguageModel
import torch
import re
from datasets import load_dataset
from tqdm import tqdm, trange
from sklearn.metrics import balanced_accuracy_score
from vllm import SamplingParams

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

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

# Replace with out specific template:
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template


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




dataset = load_dataset("achandlr/FactualConsistencyScoresTextSummarization", split='train')
dataset = dataset.select(range(1000)).shuffle(seed=3407)

predictions = []
true_labels = []

def extract_answer(response):
        answer_patterns = [
        r'answer:\s*\*?\*?(consistent|inconsistent)\*?\*?',  # Handle **consistent** format
        r'conclusion:\s*\*?\*?(consistent|inconsistent)\*?\*?', 
        r'final answer:\s*\*?\*?(consistent|inconsistent)\*?\*?',
        r'therefore.*?(consistent|inconsistent)',
        r'(consistent|inconsistent)(?:\s*[.]?\s*$)',  # At the end
    ]
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

with trange(len(dataset)) as t:
    for i in t:
        item = dataset[i]
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"""Decide if the following summary is consistent with the corresponding article.
          Note that consistency means all information in the summary is supported by the article.
          Explain your reasoning step by step first, and then answer (consistent or inconsistent) at the end:
          <Article>
        {item['context']}
          </Article>

          <Summary>
          {item['summary']}
          </Summary>

          Answer:"""},
        ]
        inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
        ).to(model.device.type)
    
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
        
        answer = extract_answer(response)
        dataset[i]['model_answer'] = answer
        print(f"Answer for item {i}: {answer} True Label: {item['label']}")
        true_labels.append(item['label'])
        predictions.append(1 if answer == 'consistent' else 0)
        if i % 5 == 0 and i > 0:
            t.set_postfix(acc=balanced_accuracy_score(true_labels, predictions))
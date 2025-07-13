from datasets import load_dataset
from openai import OpenAI
from together import Together
import re
import os
from tqdm import tqdm, trange
from template import CONSISTENCY_COT_PROMPT
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def load_dataset_from_dir(path, type= 'json', split='train'):
    """
    Load a dataset from a local directory.

    Args:
        path (str): The path to the dataset directory.
        split (str): The dataset split to load (default is 'train').

    Returns:
        Dataset: The loaded dataset.
    """
    dataset = load_dataset(type, data_files=path, split=split)
    dataset = dataset.shuffle(seed=42)
    return dataset


  
def extract_answer_qwen(text):
  pattern1 = r'\bAnswer:\sconsistent\b'
  pattern2 = r'\bFinal\sAnswer:\sconsistent\b'
  pattern3 = r'\bAnswer:\s\*\*consistent\*\*'
  pattern4 = r'\*\*Final\sAnswer:\sconsistent\*\*'
  pattern5 = r'\*\*Final\sanswer:\sconsistent\*\*'
  pattern6 = r'\bAnswer:\s\*\*Consistent\*\*'
  pattern7 = r'\b\*\*Answer:\*\*\n\*\*Consistent\*\*'
  pattern8 = r'\*\*Final\sanswer:\sConsistent\*\*'
  pattern9 = r'\bAnswer:\nconsistent'
  pattern10 = r'\bAnswer:\nConsistent'
  pattern11 = r'\*\*Answer:\*\*\s\*\*Consistent\*\*'
  pattern12 = r'\*\*Answer:\*\*\sConsistent'
  pattern12 = r'\*\*Answer:\*\*\sconsistent'
  pattern13 = r'\*\*Answer:\sconsistent\*\*'
  pattern14 = r'\*\*Answer:\sConsistent\*\*'
  pattern15 = r'Answer:\s\s\n\*\*Consistent\*\*'
  pattern16 = r'(\*\*(c|C)onsistency\*\*){1}$'
  pattern17 = r'(\b(c|C)onsistent){1}$'
  pattern18 = r'(\*\*(c|C)onsistent)\*\*{1}$'
  pattern19 = r'\bAnswer:\sconsistent\b'
  pattern20 = r'\bFinal\sAnswer:\sconsistent\b'
  pattern21 = r'\bAnswer:\s\*\*consistent\*\*'
  pattern22 = r'\*\*Final\sAnswer:\sconsistent\*\*'
  pattern23 = r'\*\*Final\sanswer:\sconsistent\*\*'
  pattern24 = r'\bAnswer:\s\*\*Consistent\*\*'
  pattern25 = r'\b\*\*Answer:\*\*\n\*\*Consistent\*\*'
  pattern26 = r'\*\*Final\sanswer:\sConsistent\*\*'
  pattern27 = r'\bAnswer:\nconsistent'
  pattern28 = r'\bAnswer:\nConsistent'
  pattern29 = r'\*\*Answer:\*\*\s\*\*Consistent\*\*'
  pattern30 = r'\*\*Answer:\*\*\sConsistent'
  pattern31 = r'\*\*Answer:\*\*\sconsistent'
  pattern32 = r'\*\*Answer:\sconsistent\*\*'
  pattern33 = r'\*\*Answer:\sConsistent\*\*'
  pattern34 = r'Answer:\s\s\n\*\*Consistent\*\*'
  pattern35 = r'(\*\*(c|C)onsistency\*\*){1}$'
  pattern36 = re.compile(
    r'^\*\*Answer\*\*:\s*Consistent\.\s*\Z',  # \Z = absolute end of string
    re.MULTILINE | re.IGNORECASE
)

  if re.search(pattern1, text) or re.search(pattern2, text) or re.search(pattern3, text)or re.search(pattern4, text)\
  or re.search(pattern5, text) or re.search(pattern6, text) or re.search(pattern7, text) or re.search(pattern9, text)\
  or re.search(pattern10, text) or re.search(pattern11, text) or re.search(pattern12, text) or re.search(pattern13, text)\
  or re.search(pattern14, text) or re.search(pattern15, text) or re.search(pattern16, text) or re.search(pattern17, text)\
  or re.search(pattern18, text) or re.search(pattern19, text) or re.search(pattern20, text) or re.search(pattern21, text)or re.search(pattern22, text)\
  or re.search(pattern23, text) or re.search(pattern24, text) or re.search(pattern25, text) or re.search(pattern26, text)\
  or re.search(pattern27, text) or re.search(pattern28, text) or re.search(pattern29, text) or re.search(pattern30, text)\
  or re.search(pattern31, text) or re.search(pattern32, text) or re.search(pattern33, text) or re.search(pattern34, text)\
  or re.search(pattern35, text) or re.search(pattern36, text):
    return 1
  else:
    return 0
  
def initialize_clients(name='gpt'):
  gpt_client = OpenAI(api_key=os.getenv('gpt_api_key'))
  dp_client = OpenAI(api_key=os.getenv('dp_api_key'), base_url="https://api.deepseek.com")
  lg_client = Together(api_key=os.getenv('tohether_api_key'))
  llama_client = Together(api_key=os.getenv('tohether_api_key'))
  qwen_client = OpenAI(
    # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
    api_key="sk-cfe35a9b04034f8597ce1fd45b8bf45c", 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
  if name == 'gpt':
    return gpt_client
  elif name == 'dp':
    return dp_client
  elif name == 'lg':
    return lg_client
  elif name == 'llama':
    return llama_client
  else:
    return qwen_client
extract_answer_qwen("""
1. **Identify the key claim in the summary:**  
   The summary states, *"the bolton boxer had hoped to take [on] floyd mayweather in a $300million mega-fight."*

2. **Determine who "the Bolton boxer" refers to:**  
   From the article, it's clear that "the Bolton boxer" is Amir Khan.

3. **Check if the article supports that Amir Khan wanted to fight Floyd Mayweather:**  
   Yes, the article says: *"The Bolton boxer had hoped to take on Floyd Mayweather but the pound-for-pound king will instead meet Manny Pacquaio in a $300million mega-fight on May 2."* This confirms that Amir Khan wanted to fight Mayweather.

4. **Check if the article mentions a "$300 million mega-fight":**  
   Yes, the article clearly states that Mayweather will meet Manny Pacquiao in a "$300million mega-fight on May 2."

5. **Putting it together:**  
   The summary accurately reflects that Amir Khan had hoped to fight Floyd Mayweather, and that Mayweather’s next fight was the $300 million matchup against Pacquiao. There are no inaccuracies or unsupported claims in the summary.

**Conclusion:**  
**Consistent**
""")
def most_frequent(List):
    return max(set(List), key=List.count)

def consistency_evaluator_doctype(dataset, client, model_name='qwen'):
   predictions = []
   true_labels = []
   with trange(len(dataset)) as t:
      for i in t:
         response = client.chat.completions.create(
            model=model_name,
            messages=[
               {"role": "system", "content": "You are a helpful assistant"},
               {"role": "user", "content": CONSISTENCY_COT_PROMPT.format(article=dataset[i]['document'], summary=dataset[i]['claim'])},
            ],
            stream=False
         )
         
         prediction = extract_answer_qwen(response.choices[0].message.content)
         predictions.append(prediction)
         true_labels.append(dataset[i]['label'])
         print(response.choices[0].message.content)
         print('-'*100)
         print(f"""Prediction: {prediction} True Label: {dataset[i]['label']}""")
         if i%5 == 0 and i > 0:
            t.set_postfix(accuracy=balanced_accuracy_score(predictions, true_labels))
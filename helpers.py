from datasets import load_dataset
from openai import OpenAI
from together import Together
from dotenv import load_dotenv
import re
import os
from tqdm import tqdm, trange
from template import CONSISTENCY_COT_PROMPT, RANKING_PROMPT
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr

# Load environment variables
load_dotenv()

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
    api_key=os.getenv('qwen_api_key'), 
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
   print(f"Final Accuracy: {accuracy_score(predictions, true_labels)}")
   print(f"Final Balanced Accuracy: {balanced_accuracy_score(predictions, true_labels)}")


def ranking_evaluator(dataset, client, model_name='deepseek-chat'):
   predictions = []
   model_names = []
   hashes = []
   with trange(len(dataset)) as t:
      for i in t:
         messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": RANKING_PROMPT.format(article=dataset[i]['article'], summary=dataset[i]['summary'])}
         ]
         response = client.chat.completions.create(model=model_name, messages=messages, stream=False)
         print(response.choices[0].message.content)
         predictions.append(np.float64(response.choices[0].message.content))
         model_names.append(dataset[i]['model_name'])
         hashes.append(dataset[i]['hash'])
         print("-"*100)
   human_annot = load_dataset('json', data_files='data/human_annotations.json', split='train')
   human_df = pd.DataFrame(human_annot)
   predictions_df = pd.DataFrame({
      'hash': hashes,
      'model_name': model_names,
      'prediction_score': predictions
   })
   human_labels_df = human_df[['hash', 'model_name', 'Factuality']]
   merged_df = pd.merge(predictions_df, human_labels_df, on=['hash', 'model_name'])
   analysis_df = merged_df.dropna(subset=['prediction_score', 'Factuality'])

   llm_scores = analysis_df['prediction_score']
   human_scores = analysis_df['Factuality']
   pearson_corr, pearson_p_value = pearsonr(llm_scores, human_scores)
   spearman_corr, spearman_p_value = spearmanr(llm_scores, human_scores)
   print(f"Pearson Correlation (ρ): {pearson_corr:.4f}")
   print(f"Spearman Correlation (r): {spearman_corr:.4f}")

def bartscore_eval(dataset):
   metrics = ['coherence', 'consistency', 'fluency', 'relevance']
   averages = {metric: [] for metric in metrics}

   for item in dataset:
      annotations = item.get('expert_annotations', [])
      if not annotations:  
         for metric in metrics:
               averages[metric].append(None) 
         continue

      num_annotations = len(annotations)
      for metric in metrics:
         total_score = sum(anno[metric] for anno in annotations)
         averages[metric].append(total_score / num_annotations)
   bartscores = load_dataset('json', data_files='/data/bartscore_results.jsonl', split='train')
   bartscores_list = [item['average_bartscore'] for item in bartscores]
   print(f"The spearman correlation between BARTScore and human scores is: {spearmanr(bartscores_list, human_scores)}")
   print(f"The pearson correlation between BARTScore and human scores is: {pearsonr(bartscores_list, human_scores)}")

class NERConsistencyEvaluator:
  def __init__(self, model_name: str="en_core_web_sm"):
    try:
      self.nlp = spacy.load(model_name)
    except OSError:
      print(f"Model {model_name} not found. Please download it with.")
      print(f"python -m spacy download {model_name}")
    self.entity_types = {
        'PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'FAC', 'PRODUCT',
        'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL',
        'ORDINAL'
    }
  def extract_entities(self, text: str) -> Dict[str, Set[str]]:
    doc = self.nlp(text)
    entities = defaultdict(set)
    for ent in doc.ents:
      if ent.label_ in self.entity_types:
        normalized_text = ent.text.lower().strip()
        if len(normalized_text) > 1:
          entities[ent.label_].add(normalized_text)
    return dict(entities)
  def get_all_entities(self, entities_dict: Dict[str, Set[str]]) -> Set[str]:
    all_entities = set()
    for entity_set in entities_dict.values():
      all_entities.update(entity_set)
    return all_entities
  def entity_overlap_score(self, document: str, summary: str) -> float:
    """
    Calculate entity overlap score: (entities in summary that are in document) / (total entities in summary)
    """
    doc_entities = self.get_all_entities(self.extract_entities(document))
    sum_entities = self.get_all_entities(self.extract_entities(summary))
    if not sum_entities:
      return 1.0
    overlap = len(sum_entities.intersection(doc_entities))
    return overlap / len(sum_entities)
  def entity_hallucination_score(self, document: str, summary: str) -> float:
    """
    Calculate entity hallucination score: (entities in summary not in the document) / (total entities in summary)
    """
    return 1.0 - self.entity_overlap_score(document, summary)
  def entity_coverage_score(self, document: str, summary: str) -> float:
    """
    Calculate entity coverage score: (document entities covered in summary) / (total entities in document)
    """
    doc_entities = self.get_all_entities(self.extract_entities(document))
    sum_entities = self.get_all_entities(self.extract_entities(summary))
    if not doc_entities:
      return 1.0
    coverage = len(doc_entities.intersection(sum_entities))
    return coverage / len(doc_entities)
  def entity_type_consistency_score(self, document: str, summary: str) -> float:
    """
    Check if entities maintain consistent types between document and summary
    """
    doc_entities = self.extract_entities(document)
    sum_entities = self.extract_entities(summary)
    doc_entity_types = {}
    for ent_type, entities in doc_entities.items():
      for entity in entities:
        doc_entity_types['entity'] = ent_type
    sum_entity_types = {}
    for ent_type, entities in sum_entities.items():
      for entity in entities:
        sum_entity_types['entity'] = ent_type
    shared_entities = set(doc_entity_types.keys()).intersection(set(sum_entity_types.keys()))

    if not shared_entities:
      return 1.0
    consistent_count = 0
    for entity in shared_entities:
      if doc_entity_types[entity] == sum_entity_types[entity]:
        consistent_count += 1
    return consistent_count / len(shared_entities)
  def comprehensive_ner_score(self, document: str, summary: str, weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Calculate the comprehensive NER Score based on multiple metrics
    """
    if weights is None:
      weights = {
        'overlap': 0.4,
        'hallucination': 0.3,
        'coverage': 0.2,
        'type_consistency': 0.1
      }
    scores = {}
    scores['entity_overlap'] = self.entity_overlap_score(document, summary)
    scores['entity_hallucination'] = self.entity_hallucination_score(document, summary)
    scores['entity_coverage'] = self.entity_coverage_score(document, summary)
    scores['entity_type_consistency'] = self.entity_type_consistency_score(document, summary)
      
    combined_score = (
          weights['overlap'] * scores['entity_overlap'] +
          weights['hallucination'] * scores['entity_hallucination'] +
          weights['coverage'] * scores['entity_coverage'] + 
          weights['type_consistency'] * scores['entity_type_consistency']
    )
    scores['combined_ner_score'] = combined_score
    return scores
  def predict_consistency(self, document: str, summary: str, threshold: float = 0.7) -> int:
    overlap_score = self.entity_overlap_score(document, summary)
    return 1 if overlap_score >= threshold else 0 

def evaluate_ner_on_factcc_dataset(dataset, output_file='ner_results.csv'):
      """
      Evaluate NER overlap method on FactCC-style dataset.
      
      Args:
         dataset: Dataset with 'text' (document) and 'claim' (summary) columns
         output_file: File to save results
      """
      print("Initializing NER Consistency Evaluator...")
      try:
         evaluator = NERConsistencyEvaluator()
      except:
         print("Error: Please install spaCy and download the English model:")
         print("pip install spacy")
         print("python -m spacy download en_core_web_sm")
         return None
      
      print(f"Evaluating {len(dataset)} examples...")
      
      results = {
         'overlap_scores': [],
         'hallucination_scores': [],
         'coverage_scores': [],
         'type_consistency_scores': [],
         'combined_scores': [],
         'predictions': [],
         'true_labels': []
      }
      # Different thresholds to test
      thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
      threshold_results = {f'pred_{t}': [] for t in thresholds}
      with trange(len(dataset)) as t:
         for i in t:
               # Get document and summary
               if hasattr(dataset[i], 'text'):
                  document = dataset[i]['text']
                  summary = dataset[i]['claim']
                  true_label = 1 if dataset[i]['label'] == 'CORRECT' else 0
               else:
                  # Handle different dataset formats
                  document = dataset[i].get('document', dataset[i].get('text', ''))
                  summary = dataset[i].get('summary', dataset[i].get('claim', ''))
                  true_label = dataset[i].get('consistency', dataset[i].get('label', 0))
                  if isinstance(true_label, str):
                     true_label = 1 if true_label == 'CORRECT' else 0
               
               # Calculate comprehensive NER scores
               scores = evaluator.comprehensive_ner_score(document, summary)
               results['overlap_scores'].append(scores['entity_overlap'])
               results['hallucination_scores'].append(scores['entity_hallucination'])
               results['coverage_scores'].append(scores['entity_coverage'])
               results['type_consistency_scores'].append(scores['entity_type_consistency'])
               results['combined_scores'].append(scores['combined_ner_score'])
               results['true_labels'].append(true_label)
               for threshold in thresholds:
                  pred = evaluator.predict_consistency(document, summary, threshold)
                  threshold_results[f'pred_{threshold}'].append(pred)
               if i % 10 == 0 and i > 0:
                  current_acc = accuracy_score(
                     results['true_labels'], 
                     threshold_results['pred_0.7'][:len(results['true_labels'])]
                  )
                  t.set_postfix(accuracy=f"{current_acc:.3f}")
      print("\nNER Overlap Method Results:")
      print("=" * 50)
      
      best_threshold = 0.7
      best_f1 = 0
      for threshold in thresholds:
         predictions = threshold_results[f'pred_{threshold}']
         accuracy = accuracy_score(results['true_labels'], predictions)
         balanced_acc = balanced_accuracy_score(results['true_labels'], predictions)
         precision, recall, f1, _ = precision_recall_fscore_support(
               results['true_labels'], predictions, average='binary'
         )
         
         print(f"\nThreshold {threshold}:")
         print(f"  Accuracy: {accuracy:.3f}")
         print(f"  Balanced Accuracy: {balanced_acc:.3f}")
         print(f"  Precision: {precision:.3f}")
         print(f"  Recall: {recall:.3f}")
         print(f"  F1-Score: {f1:.3f}")
         
         if f1 > best_f1:
               best_f1 = f1
               best_threshold = threshold
      print(f"\nBest Threshold: {best_threshold} (F1: {best_f1:.3f})")

      # Correlation analysis with human annotations (if available)
      print("\nScore Statistics:")
      print(f"  Mean Overlap Score: {np.mean(results['overlap_scores']):.3f}")
      print(f"  Mean Hallucination Score: {np.mean(results['hallucination_scores']):.3f}")
      print(f"  Mean Coverage Score: {np.mean(results['coverage_scores']):.3f}")
      print(f"  Mean Combined Score: {np.mean(results['combined_scores']):.3f}")
      
      # Save results
      results_df = pd.DataFrame({
         'overlap_score': results['overlap_scores'],
         'hallucination_score': results['hallucination_scores'],
         'coverage_score': results['coverage_scores'],
         'type_consistency_score': results['type_consistency_scores'],
         'combined_score': results['combined_scores'],
         'true_label': results['true_labels'],
         **threshold_results
      })
      
      results_df.to_csv(output_file, index=False)
      print(f"\nResults saved to {output_file}")
      
      return results_df, best_threshold
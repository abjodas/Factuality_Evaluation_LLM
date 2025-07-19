from helpers import load_dataset_from_dir, initialize_clients, consistency_evaluator_doctype, ranking_evaluator, bartscore_eval, NERConsistencyEvaluator
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser(description="arguments for evaluatiing factuality")
parser.add_argument("--dataset_name", type=str, default="cogensumm", help="Name of the dataset to evaluate(cogensumm, factcc, polytope, summeval, xsumfaith, frank)")
parser.add_argument("--llm_provider", type=str, default="dp", help="Name of the model to use for evaluation(qwen, gpt, dp, lg, llama)")
parser.add_argument("--model_name", type=str, default="deepseek-chat", help="Name of the model to use for evaluation")
parser.add_argument("--task", type=str, default="consistency", help="Task to evaluate (e.g., consistency, ranking, )")
parser.add_argument("--split", type=str, default='val', help="Split of the dataset to use for evaluation (e.g., train, val, test)")
args = parser.parse_args()



if __name__ == "__main__":
    if (args.dataset_name == "cogensumm" or args.dataset_name == "factcc" or args.dataset_name == "polytope" or args.dataset_name == "summeval" or args.dataset_name == "xsumfaith") and args.task == "consistency":
        dataset = load_dataset_from_dir(f"data/{args.dataset_name}_{args.split}.jsonl", type='json', split='train')
        consistency_evaluator_doctype(dataset, client=initialize_clients(args.llm_provider), model_name=args.model_name)
    elif args.task == "ranking" or args.dataset_name == "frank":
        dataset = load_dataset_from_dir(f"data/benchmark_data.json", type='json', split='train')
        ranking_evaluator(dataset, client=initialize_clients(args.llm_provider), model_name=args.model_name)
    elif args.task == "bartscore":
        dataset = load_dataset_from_dir(f"data/{"human_annotations.aligned.paired.jsonl"}.jsonl", type='json', split='train')
        bartscore_eval(dataset)
    elif args.task == "ner_consistency":
        if args.dataset_name == 'factcc':
            dataset = load_dataset("mtc/factcc_annotated_eval_data")
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
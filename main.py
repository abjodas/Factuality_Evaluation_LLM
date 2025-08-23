from helpers import evaluate_ranking_consistency, evaluate_ranking_consistency_summac, load_dataset_from_dir, initialize_clients, consistency_evaluator_doctype, ranking_evaluator, bartscore_eval, evaluate_ner_on_factcc_dataset, evaluate_additional_metrics, evaluate_correlation_llm, consistency_evaluator_factcc
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser(description="arguments for evaluatiing factuality")
parser.add_argument("--dataset_name", type=str, default="cogensumm", help="Name of the dataset to evaluate(cogensumm, factcc, polytope, summeval, xsumfaith, frank)")
parser.add_argument("--llm_provider", type=str, default="dp", help="Name of the model to use for evaluation(qwen, gpt, dp, lg, llama)")
parser.add_argument("--trad_method", type=str, default="", help="Name of the traditional method to use for evaluation (summac, bartscore, ner_consistency)")
parser.add_argument("--model_name", type=str, default="deepseek-chat", help="Name of the model to use for evaluation")
parser.add_argument("--task", type=str, default="consistency", help="Task to evaluate (e.g., consistency, ranking, )")
parser.add_argument("--split", type=str, default='val', help="Split of the dataset to use for evaluation (e.g., train, val, test)")
parser.add_argument('--type', type=str, default='COT', help='Type of evaluation to perform (COT, no_COT)')
args = parser.parse_args()



if __name__ == "__main__":
    if (args.dataset_name == "cogensumm") and args.task == "consistency":
        dataset = load_dataset_from_dir(f"data/{args.dataset_name}_{args.split}.jsonl", type='json', split='train')
        consistency_evaluator_doctype(dataset, client=initialize_clients(args.llm_provider), model_name=args.model_name)
    elif args.dataset_name == 'factcc' and args.task == 'consistency':
        dataset = load_dataset("mtc/factcc_annotated_eval_data", split=args.split)
        consistency_evaluator_factcc(dataset, client=initialize_clients(args.llm_provider), model_name=args.model_name)
    elif args.task == "ranking" and args.dataset_name == "frank":
        dataset = load_dataset_from_dir(f"data/benchmark_data.json", type='json', split='train')
        ranking_evaluator(dataset, client=initialize_clients(args.llm_provider), model_name=args.model_name)
    elif args.task == "bartscore":
        dataset = load_dataset_from_dir(f"data/{"human_annotations.aligned.paired.jsonl"}.jsonl", type='json', split='train')
        bartscore_eval(dataset)
    elif args.task == "ner_consistency":
        if args.dataset_name == 'factcc':
            dataset = load_dataset("mtc/factcc_annotated_eval_data")
            results_df, best_threshold = evaluate_ner_on_factcc_dataset(dataset)
            print(f"Best threshold: {best_threshold}")
            print(results_df)
        if args.dataset_name == 'polytope':
            dataset = load_dataset_from_dir(f'data/polytope_{args.split}.jsonl', type='json', split='train')
            results_df, best_threshold = evaluate_ner_on_factcc_dataset(dataset)
            print(f"Best threshold: {best_threshold}")
            print(results_df)
    elif args.task == 'correlation':
        dataset = load_dataset_from_dir("data/model_annotations.aligned.paired.jsonl")
        evaluate_additional_metrics(dataset)
    elif args.task == 'correlation_llm':
        print("Correlation LLm")
        dataset = load_dataset_from_dir("data/model_annotations.aligned.paired.jsonl")
        results_df = evaluate_correlation_llm(dataset, model_name=args.model_name, llm_provider=args.llm_provider, type=args.type)
    elif args.task == 'ranking' and args.dataset_name == "fib" and len(args.trad_method) > 0 and args.trad_method == "summac":
        dataset = load_dataset("r-three/fib", split='test')
        dataset = dataset.shuffle(seed=32).select(range(600))
        results_df = evaluate_ranking_consistency_summac(dataset, model_name=args.model_name, llm_provider=args.llm_provider, output_file='fib_ranking_results.csv')
    elif args.task == 'ranking' and args.dataset_name == "fib":
        dataset = load_dataset("r-three/fib", split='test')
        dataset = dataset.shuffle(seed=32).select(range(600))
        results_df = evaluate_ranking_consistency(dataset, model_name=args.model_name, llm_provider=args.llm_provider, output_file='fib_ranking_results.csv', type=args.type)

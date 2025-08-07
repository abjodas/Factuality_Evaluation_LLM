from helpers import load_dataset_from_dir, initialize_clients, consistency_evaluator_doctype, ranking_evaluator, bartscore_eval, evaluate_ner_on_factcc_dataset, evaluate_additional_metrics
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


from helpers import load_dataset_from_dir, extract_answer_qwen, initialize_clients, consistency_evaluator_doctype
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import argparse

parser = argparse.ArgumentParser(description="arguments for evaluatiing factuality")
parser.add_argument("--dataset_name", type=str, default="cogensumm", help="Name of the dataset to evaluate")
parser.add_argument("--llm_provider", type=str, default="qwen", help="Name of the model to use for evaluation(qwen, gpt, dp, lg, llama)")
parser.add_argument("--model_name", type=str, default="deepseek-chat", help="Name of the model to use for evaluation")
parser.add_argument("--task", type=str, default="consistency", help="Task to evaluate (e.g., consistency, ranking, )")
args = parser.parse_args()

print(f"Loading dataset from {args.dataset_path}")


if __name__ == "__main__":
    if args.dataset_name == "cogensumm":
        dataset = load_dataset_from_dir("data/cogensumm", type='json', split='test')
    if args.task == "consistency" and args.dataset_name == "cogensumm":
        consistency_evaluator_doctype(dataset, client=initialize_clients(args.llm_provider), model_name=args.model_name)

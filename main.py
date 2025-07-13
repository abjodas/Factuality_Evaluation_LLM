from helpers import load_dataset_from_dir, extract_answer_qwen, initialize_clients, consistency_evaluator_doctype
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import argparse

parser = argparse.ArgumentParser(description="arguments for evaluatiing factuality")
parser.add_argument("--dataset_name", type=str, default="cogensumm", help="Name of the dataset to evaluate(cogensumm, factcc, polytope, summeval, xsumfaith, frank)")
parser.add_argument("--llm_provider", type=str, default="dp", help="Name of the model to use for evaluation(qwen, gpt, dp, lg, llama)")
parser.add_argument("--model_name", type=str, default="deepseek-chat", help="Name of the model to use for evaluation")
parser.add_argument("--task", type=str, default="consistency", help="Task to evaluate (e.g., consistency, ranking, )")
parser.add_argument("--split", type=str, default='val', help="Split of the dataset to use for evaluation (e.g., train, val, test)")
args = parser.parse_args()



if __name__ == "__main__":
    if args.dataset_name == "cogensumm" or args.dataset_name == "factcc" or args.dataset_name == "polytope" or args.dataset_name == "summeval" or args.dataset_name == "xsumfaith":
        dataset = load_dataset_from_dir(f"data/{args.dataset_name}_{args.split}.jsonl", type='json', split='train')
    if args.task == "consistency" and (args.dataset_name == "cogensumm" or args.dataset_name == "factcc" or args.dataset_name == "polytope" or args.dataset_name == "summeval" or args.dataset_name == "xsumfaith"):
        consistency_evaluator_doctype(dataset, client=initialize_clients(args.llm_provider), model_name=args.model_name)

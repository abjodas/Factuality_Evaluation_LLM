from helpers import load_dataset_from_dir, extract_answer_qwen, initialize_clients
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import argparse

parser = argparse.ArgumentParser(description="arguments for evaluatiing factuality")
parser.add_argument("--dataset_path", type=str, default="data/processed/processed_dataset.json", help="Path to the dataset")
parser.add_argument("--model_name", type=str, default="qwen", help="Name of the model to use for evaluation(qwen, gpt, dp, lg, llama)")
parser.add_argument("--task", type=str, default="consistency", help="Task to evaluate (e.g., consistency, ranking, )")
args = parser.parse_args()

print(f"Loading dataset from {args.dataset_path}")


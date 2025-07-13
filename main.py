from helpers import load_dataset_from_dir, extract_answer_qwen, initialize_clients
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import argparse

parser = argparse.ArgumentParser(description="arguments for evaluatiing factuality")
parser.add_argument("--dataset_path", type=str, default="data/processed/processed_dataset.json", help="Path to the dataset")
parser.add_argument("--model_name", type=str, default="qwen", help="Name of the model to use for evaluation(qwen, gpt, dp, lg, llama)")
parser.add_argument("--task", type=str, default="consistency", help="Task to evaluate (e.g., consistency, ranking, )")
args = parser.parse_args()

print(f"Loading dataset from {args.dataset_path}")

if __name__ == "__main__":
    dataset = load_dataset_from_dir(args.dataset_path)
    print(f"Loaded {len(dataset)} samples from the dataset")

    if args.model_name == "qwen":
        client = initialize_clients(model_name=args.model_name)
        answers = extract_answer_qwen(dataset, client, task=args.task)
    else:
        raise ValueError(f"Model {args.model_name} is not supported for evaluation.")

    # Assuming answers is a list of predictions and dataset has ground truth labels
    y_true = [sample['label'] for sample in dataset]
    y_pred = [answer['prediction'] for answer in answers]

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
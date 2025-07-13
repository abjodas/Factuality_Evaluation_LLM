from helpers import load_dataset_from_dir, extract_answer_qwen, initialize_clients
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import argparse

parser = argparse.ArgumentParser(description="arguments for evaluatiing factuality")
parser.add_argument("--dataset_path", type=str, default="data/processed/processed_dataset.json", help="Path to the dataset")
parser.add_argument("--model_name", type=str, default="qwen", help="Name of the model to use for evaluation(qwen, gpt, dp, lg, llama)")
parser.add_argument("--task", type=str, default="consistency", help="Task to evaluate (e.g., consistency, ranking, )")
args = parser.parse_args()

print(f"Loading dataset from {args.dataset_path}")

if args.task == "consistency":
    dataset = load_dataset_from_dir(args.dataset_path)
    print(f"Loaded {len(dataset)} samples for consistency evaluation.")
    
    clients = initialize_clients(args.model_name)
    print(f"Initialized clients for model: {args.model_name}")

    predictions = []
    for sample in dataset:
        answer = extract_answer_qwen(sample, clients)
        predictions.append(answer)

    # Assuming the dataset has a 'label' field for ground truth
    ground_truth = [sample['label'] for sample in dataset]
    
    accuracy = accuracy_score(ground_truth, predictions)
    balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
import os

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from datasets import load_from_disk


def process_on_gpu(
    gpu_id: int, dataset_path: str, model_name_or_path: str, output_path: str, k: int, batch_size: int, num_gpus: int
) -> None:
    """Process dataset on a specific GPU."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Load dataset
    print(f"GPU {gpu_id}: Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)

    # Split dataset for this GPU
    total_size = len(dataset)
    chunk_size = total_size // num_gpus
    start_idx = gpu_id * chunk_size
    end_idx = start_idx + chunk_size if gpu_id < num_gpus - 1 else total_size
    dataset = dataset.select(range(start_idx, end_idx))

    # Load model
    print(f"GPU {gpu_id}: Loading model from {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()

    def process_batch(examples: dict[str, list]) -> dict[str, list]:
        """Process a batch of examples through the model and extract top k logits."""
        # Convert input_ids to tensors and move to device
        seq_len = [len(e) for e in examples["input_ids"]]
        max_seq_len = max(seq_len)
        input_ids = [exp + [0] * (max_seq_len - len(exp)) for exp in examples["input_ids"]]
        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.ones_like(input_ids)  # Since it's already packed, all tokens are valid

        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = torch.nn.functional.log_softmax(outputs.logits.to(torch.float32), dim=-1)
            top_k_values, top_k_indices = torch.topk(logits, k, largest=True, dim=-1)

        # Convert to lists for storage
        batch_top_k_logits = top_k_values.cpu().numpy().tolist()
        batch_top_k_logits = [
            logits[: len(e)] for logits, e in zip(batch_top_k_logits, examples["input_ids"], strict=False)
        ]
        batch_top_k_indices = top_k_indices.cpu().numpy().tolist()
        batch_top_k_indices = [
            indices[: len(e)] for indices, e in zip(batch_top_k_indices, examples["input_ids"], strict=False)
        ]

        # Create a new dictionary with the original input_ids and the top k logits and indices
        new_examples = {
            "input_ids": examples["input_ids"],
            "top_k_logits": batch_top_k_logits,
            "top_k_indices": batch_top_k_indices,
        }

        return new_examples

    # Process the dataset in batches
    print(f"GPU {gpu_id}: Processing dataset through model...")
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=[col for col in dataset.column_names if col != "input_ids"],  # Keep input_ids
    )

    # Save the processed dataset
    gpu_output_path = f"{output_path}_gpu{gpu_id}"
    print(f"GPU {gpu_id}: Saving processed dataset to {gpu_output_path}")
    processed_dataset.save_to_disk(gpu_output_path)
    print(f"GPU {gpu_id}: Done!")


def extract_top_k_logits(
    dataset_path: str, model_name_or_path: str, output_path: str, k: int = 5, batch_size: int = 32, num_gpus: int = 2
) -> None:
    """
    Load a tokenized dataset from disk, process it through a model using multiple GPUs,
    and save the top k logits for each token.

    Args:
        dataset_path (str): Path to the tokenized dataset on disk
        model_name_or_path (str): Name or path of the model to use
        output_path (str): Path to save the processed dataset
        k (int): Number of top logits to save for each token
        batch_size (int): Batch size for processing
        num_gpus (int): Number of GPUs to use
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA to be available")

    if torch.cuda.device_count() < num_gpus:
        raise RuntimeError(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} are available")

    # Set the start method to 'spawn' for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    # Create processes for each GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=process_on_gpu,
            args=(gpu_id, dataset_path, model_name_or_path, output_path, k, batch_size, num_gpus),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Combine the results from all GPUs
    print("Combining results from all GPUs...")
    from datasets import concatenate_datasets

    datasets = []
    for gpu_id in range(num_gpus):
        gpu_output_path = f"{output_path}_gpu{gpu_id}"
        datasets.append(load_from_disk(gpu_output_path))

    combined_dataset = concatenate_datasets(datasets)
    combined_dataset.save_to_disk(output_path)

    # Clean up temporary GPU-specific datasets
    for gpu_id in range(num_gpus):
        gpu_output_path = f"{output_path}_gpu{gpu_id}"
        if os.path.exists(gpu_output_path):
            import shutil

            shutil.rmtree(gpu_output_path)

    print("All done!")


if __name__ == "__main__":
    # Example usage
    extract_top_k_logits(
        dataset_path="/work/benyamin/trl-experiment-template/datasets/FineWeb-edu/sample-10BT/train_dataset",
        model_name_or_path="HuggingFaceTB/SmolLM-1.7B",
        output_path="/work/benyamin/trl-experiment-template/datasets/FineWeb-edu/sample-10BT-kd",
        k=5,
        num_gpus=6,
    )

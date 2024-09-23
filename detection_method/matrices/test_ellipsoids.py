"""
    Compute the mean matrix and the standard deviation matrix.
"""

import os
import json
import torch


def compute_statistics(matrix_path: str) -> dict[str, dict[str, list]]:
    # Load matrices and compute mean matrix and standard deviation matrix
    statistics: dict[str, dict[str, list]] = {}
    for i in range(10):  # TODO: 10 classes
        matrices = [torch.load(f"{matrix_path}{i}/{file}") for file in os.listdir(f"{matrix_path}{i}/")]
        # Stack all matrices to compute statistics across all matrices in a subfolder
        stacked_matrices = torch.stack(matrices)
        # Compute mean and std across the stacked matrices
        mean_matrix = torch.mean(stacked_matrices, dim=0)
        std_matrix = torch.std(stacked_matrices, dim=0)
        statistics[i] = {'mean': mean_matrix, 'std': std_matrix}
    return statistics


def compute_train_statistics(path, max_epoch: int) -> None:
    # Compute statistics for each epoch
    print("Computing matrix statistics.")
    for i in range(0,max_epoch+1,5):
        matrices_path = f'{path}epoch{i}/'
        if not os.path.exists(matrices_path):
            continue
        print(f"Epoch {i}")

        statistics = compute_statistics(matrices_path)    
        for _, stats in statistics.items():
            for key, tensor in stats.items():
                if tensor.numel() == 1:  # If the tensor has only one element, convert to a Python scalar
                    stats[key] = tensor.item()
                else:
                    stats[key] = tensor.tolist()

        with open(f"{matrices_path}matrix_statistics.json", 'w') as json_file:
            json.dump(statistics, json_file, indent=4)

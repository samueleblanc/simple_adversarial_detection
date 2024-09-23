"""
    Train a nn and compute the matrices.
"""

from detection_method.neural_nets.training import train_nn
from detection_method.neural_nets.plot_graphs import plot_graphs
from detection_method.matrices.compute_matrices import ComputeMatrices


class NN:
    def __init__(self, dataset: str, path: str, train: bool, build_matrices: bool, lr:float=0.01, bs:int=8, epochs:int=100, hidden_size:list=500, num_layers: int=5, plot:bool=False) -> None:
        self._dataset = dataset
        self._path = path if path[-1] == "/" else f"{path}/"
        self._path = f"{self._path}{dataset}/MLP-{lr}-{bs}-{num_layers}x{hidden_size}/"
        self._train = train
        self._build_matrices = build_matrices
        self._epochs = epochs
        if train:
            train_nn(dataset, self._path, lr, bs, epochs, hidden_size, num_layers)
        if plot:
            plot_graphs(self._path)
        if build_matrices:
            mat = ComputeMatrices(self._path, self._dataset, 200, hidden_size, num_layers)
            mat.values_on_epoch(epochs)
            mat.matrix_statistics(epochs, "train")
    
    def compute_matrix_statistics(self, train_or_test: str) -> None:
        mat = ComputeMatrices(self._path, self._dataset, 200)
        mat.matrix_statistics(self._epochs, train_or_test)
    
    def plot_performance(self, save: bool=True) -> None:
        plot_graphs(self._path, save)
    
    def get_dataset(self) -> str: return self._dataset
    def get_path(self) -> str: return self._path
    def get_epochs(self) -> str: return self._epochs


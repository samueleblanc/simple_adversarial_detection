import os
import sys
import tqdm
import torch
from torch.utils.data import DataLoader, Subset

from detection_method.neural_nets.model import MLP
from detection_method.neural_nets.representation import MlpRepresentation
from detection_method.neural_nets.training import get_architecture, get_dataset
from detection_method.matrices.test_ellipsoids import compute_train_statistics


class ComputeMatrices:
    def __init__(self, path: str, dataset: str, num_samples: int, hidden_size:int=500, num_layers:int=5, training_set:bool=True) -> None:
        self.path = path
        self.dataset = dataset
        self.num_samples = num_samples
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_shape = (3, 32, 32) if self.dataset == "cifar10" else (1, 28, 28)
        self.training_set = training_set
        self.device = "cpu"
        self.num_classes = 10

        print("Loading data...", flush=True)
        self.train_data, self.test_data = get_dataset(self.dataset, False)

    def compute_matrices_epoch_on_dataset(self, model: MLP, epoch: int) -> None:
        if self.training_set:
            data = self.train_data
        else:
            data = self.test_data

        print("Constructing representation", flush=True)

        if isinstance(model, MLP):
            representation = MlpRepresentation(model=model, device=self.device)
        else:
            ValueError("Architecture not supported!")

        print("Representation constructed", flush=True)
        print("Computing matrices", flush=True)

        for i in tqdm.trange(self.num_classes):
            print(f"Class {i}", flush=True, file=sys.stderr)
            train_indices = [idx for idx, target in enumerate(data.targets) if target in [i]]
            sub_train_dataloader = DataLoader(Subset(data, train_indices),
                                              batch_size=int(self.num_samples),
                                              drop_last=True)

            x_train = next(iter(sub_train_dataloader))[0] # 0 for input and 1 for label
            self.average_matrix(x_train, representation, epoch, i)

    def values_on_epoch(self, epoch:int=100) -> None:
        model_path = f"{self.path}weights/epoch{epoch}.pth"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))

        model = get_architecture(self.input_shape, self.num_classes, self.hidden_size, self.num_layers)
        model.load_state_dict(state_dict)
        self.compute_matrices_epoch_on_dataset(model, epoch)

    def average_matrix(self, data, representation, epoch: int, class_no: int, train=True) -> None:
        print(f"Average matrix. Class {class_no}. Epoch {epoch}.")
        save_dir = f"{self.path}matrices/"
        if train:
            save_dir += "train/"
        else:
            save_dir += "test/"
        save_dir += f"epoch{epoch}/{class_no}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, d in enumerate(data):
            rep = representation.forward(d)
            torch.save(rep, f"{save_dir}matrix{i}.pt")

    def matrix_statistics(self, max_epoch: int, train_or_test: str) -> None:
        compute_train_statistics(f"{self.path}matrices/{train_or_test}/", max_epoch)

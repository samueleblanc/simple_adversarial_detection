import torch

from detection_method.neural_nets.model import MLP
from detection_method.neural_nets.training import get_architecture


def get_ellipsoid_data(ellipsoids: dict, result: torch.Tensor, param: str) -> torch.Tensor:
    return torch.Tensor(ellipsoids[str(result.item())][param])

def is_in_ellipsoid(matrix: torch.Tensor, ellipsoid_mean: torch.Tensor, ellipsoid_std: torch.Tensor, width:float=2) -> torch.LongTensor:
    # Returns number of entries that are within the hyper-ellipsoid mean +- width*std
    low_bound = torch.le(ellipsoid_mean-width*ellipsoid_std, matrix)
    up_bound = torch.le(matrix, ellipsoid_mean+width*ellipsoid_std)
    return torch.count_nonzero(torch.logical_and(low_bound, up_bound))

def zero_std(matrix: torch.Tensor, ellipsoid_std: torch.Tensor, eps: float=0.1) -> torch.LongTensor:
    # Returns number of entries where it is false that this entry is close to zero in std but not in the matrix
    return torch.count_nonzero(torch.logical_and((ellipsoid_std <= eps), (matrix > eps)))  # ¬(P => Q) <==> P ∧ ¬Q

def get_model(path: str, input_shape: tuple[int]=(1,28,28), num_classes: int=10, hidden_size:int=500, num_layers:int=5) -> MLP:
    weight_path = torch.load(path, map_location=torch.device('cpu'))
    model = get_architecture(num_classes=num_classes, input_shape=input_shape, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(weight_path)
    return model

def subset(train_set, length: int, input_shape: tuple[int]) -> tuple[torch.Tensor]:
    # Returns a subset of the training data
    idx = torch.randint(low=0,high=len(train_set),size=[length],generator=torch.Generator("cpu"))
    exp_dataset = torch.zeros([length,input_shape[0],input_shape[1],input_shape[2]])
    exp_labels = torch.zeros([length],dtype=torch.long)
    for i,j in enumerate(idx):
        exp_dataset[i] = train_set[j][0]
        exp_labels[i] = train_set.targets[j]
    return (exp_dataset, exp_labels)

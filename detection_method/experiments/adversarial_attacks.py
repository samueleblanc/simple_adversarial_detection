"""
    Implementation of the detection method for adversarial examples and out-of-distribution data. 
"""

import json
import torch
import torchvision
import torchvision.transforms as transforms
import torchattacks
import numpy.random as rand

from detection_method.neural_nets.representation import MlpRepresentation
from detection_method.neural_nets.model import MLP
from detection_method.experiments.utils import get_ellipsoid_data, is_in_ellipsoid, zero_std, get_model, subset


class Experiment:

    def __init__(self, neural_net, n:int=10000, seed:int=123456) -> None:
        torch.manual_seed(seed)
        self._path = neural_net.get_path()
        self._dataset = neural_net.get_dataset()
        self._in_shape = (1,28,28) if self._dataset == "mnist" or self._dataset == "fashion" else (3,32,32)
        self._epochs = neural_net.get_epochs()
        self._weight_path = f"{self._path}weights/epoch{self._epochs}.pth"
        self._matrices_path = f"{self._path}matrices/train/epoch{self._epochs}/"

        self._n = n  # Size of subset on which to make the experiments
        
        ellipsoids_file = open(f"{self._matrices_path}matrix_statistics.json")
        self._ellipsoids: dict = json.load(ellipsoids_file)
        self._representation = MlpRepresentation(get_model(self._weight_path, input_shape=self._in_shape))
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if self._dataset == "mnist":
            train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        elif self._dataset == "fashion":
            train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
            test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
        elif self._dataset == "cifar10":
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        else:
            NotImplementedError()

        self._adv = ["GN","FGSM","RFGSM","PGD","EOTPGD"]
        self._att = {a: i for i,a in enumerate(["None"]+self._adv)}

        # Make set of images for experiment
        self._exp_dataset_train, self._exp_labels_train = subset(train_set, self._n, self._in_shape)
        self._exp_dataset_test, self._exp_labels_test = subset(test_set, self._n, self._in_shape)

    def reject_predicted_attacks(self, std_z: float=1.5, std_e:float=2, eps: float=0.1, eps_p: float=0.1, subsample_size:int=2000, standard_test = True) -> None:
        # Take a few examples from exp_dataset and compute mean and std of zero dims
        # For the other examples which are either normal or attacked, predict if it was attacked or not.
        n = subsample_size  # Sample size to compute mean and std of zero dims
        model: MLP = get_model(self._weight_path, input_shape=self._in_shape)
        attacks_cls = dict(zip(["None"]+self._adv,
                            [torchattacks.VANILA(model),
                             torchattacks.GN(model),
                             torchattacks.FGSM(model),
                             torchattacks.RFGSM(model),
                             torchattacks.PGD(model),
                             torchattacks.EOTPGD(model)
                            ]))
        attacked_dataset = {a: attacks_cls[a](self._exp_dataset_test[n:], self._exp_labels_test[n:]) for a in ["None"]+self._adv}
        # Compute mean and std of number of (almost) zero dims
        print("Compute rejection level.")
        zeros = torch.Tensor()
        in_ell = torch.Tensor()
        for im in self._exp_dataset_train[:n]:
            pred = torch.argmax(model.forward(im))
            mat = self._representation.forward(im)
            mean_mat = get_ellipsoid_data(self._ellipsoids,pred,"mean")
            std_mat = get_ellipsoid_data(self._ellipsoids,pred,"std")
            zeros = torch.cat((zeros, zero_std(mat, std_mat, eps).expand([1])))
            in_ell = torch.cat((in_ell, is_in_ellipsoid(mat, mean_mat, std_mat).expand([1])))
        zeros_lb = zeros.mean().item() - std_z*zeros.std().item()
        zeros_ub = zeros.mean().item() + std_z*zeros.std().item()
        in_ell_lb = in_ell.mean().item() - std_e*in_ell.std().item()
        in_ell_ub = in_ell.mean().item() + std_e*in_ell.std().item()
        if standard_test:
            print(f"Will reject when 'zero dims' is less than {zeros_lb}.")
        else:
            print(f"Will reject when 'zero dims' is not in [{zeros_lb}, {zeros_ub}] or when 'in ell' is not in [{in_ell_lb},{in_ell_ub}].")

        results: list[tuple[bool, bool]] = []  # (Rejected, Was attacked)
        for i in range(len(self._exp_dataset_test[n:])):
            a = rand.choice(["None"]+self._adv, p=[0.5]+[1/(2*len(self._adv)) for _ in self._adv])
            im = attacked_dataset[a][i]
            pred = torch.argmax(model.forward(im))
            mat = self._representation.forward(im)
            mean_mat = get_ellipsoid_data(self._ellipsoids,pred,"mean")
            std_mat = get_ellipsoid_data(self._ellipsoids,pred,"std")
            zero_dims = zero_std(mat, std_mat, eps_p).item()
            in_ell = is_in_ellipsoid(mat, mean_mat, std_mat).item()
            reject_zeros = (zeros_lb > zero_dims) or ((zeros_ub < zero_dims) and standard_test is False)
            reject_dims = ((in_ell_lb > in_ell) or (in_ell_ub < in_ell)) and standard_test is False
            results.append((reject_zeros or reject_dims, (a != "None")))

        good_defence = 0
        wrongly_rejected = 0
        num_att = 0
        for rej,att in results:
            if att:
                good_defence += int(rej)
                num_att += 1
            else:
                wrongly_rejected += int(rej)
        print(f"Percentage of good defences: {good_defence/num_att}.")
        print(f"Percentage of wrong rejections: {wrongly_rejected/(len(results)-num_att)}.")

    def reject_predicted_out_of_dist(self, std_z: float=1.5, std_e:float=2, eps: float=0.1, eps_p: float=0.1, delta: float=2, delta_p: float=2, subsample_size:int=2000, z_upper_bound=False) -> None:
        # Same principle as with 'reject_predicted_attacks', but with 'out of distribution' data
        n = subsample_size  # Sample size to compute mean and std of zero dims
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if self._dataset == "mnist":
            out_of_dist = "fashion"
            out_dist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
        elif self._dataset == "fashion":
            out_of_dist = "mnist"
            out_dist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        else:
            raise NotImplementedError(f"Data set must be 'mnist' or 'fashion', not '{self._dataset}'.")
        model: MLP = get_model(self._weight_path, input_shape=self._in_shape)
        out_dist_data, _ = subset(out_dist_test, len(self._exp_dataset_test), input_shape=self._in_shape)
        print(f"Model: {self._dataset}")
        print(f"Out of distribution data: {out_of_dist}")
        # Compute mean and std of number of (almost) zero dims in in_dist_data
        print("Compute rejection level.")
        zeros = torch.Tensor()
        in_ell = torch.Tensor()
        for im in self._exp_dataset_train[:n]:
            pred = torch.argmax(model.forward(im))
            mat = self._representation.forward(im)
            mean_mat = get_ellipsoid_data(self._ellipsoids,pred,"mean")
            std_mat = get_ellipsoid_data(self._ellipsoids,pred,"std")
            zeros = torch.cat((zeros, zero_std(mat, std_mat, eps).expand([1])))
            in_ell = torch.cat((in_ell, is_in_ellipsoid(mat, mean_mat, std_mat, delta).expand([1])))
        zeros_lb = zeros.mean().item() - std_z*zeros.std().item()
        zeros_ub = zeros.mean().item() + std_z*zeros.std().item()
        in_ell_lb = in_ell.mean().item() - std_e*in_ell.std().item()
        in_ell_ub = in_ell.mean().item() + std_e*in_ell.std().item()
        if z_upper_bound:
            print(f"Will reject when 'zero dims' is not in [{zeros_lb}, {zeros_ub}] or when 'in ell' is not in [{in_ell_lb},{in_ell_ub}].")
        else:
            print(f"Will reject when 'zero dims' is less than {zeros_lb} or when 'in ell' is not in [{in_ell_lb},{in_ell_ub}].")
        results: list[tuple[bool, bool]] = []  # (Rejected, Was out of dist)
        
        data = {"in dist": self._exp_dataset_test[n:], "out dist": out_dist_data[n:]}
        for i in range(len(self._exp_dataset_test[n:])):
            d = rand.choice(["in dist", "out dist"])
            im = data[d][i]
            pred = torch.argmax(model.forward(im))
            mat = self._representation.forward(im)
            mean_mat = get_ellipsoid_data(self._ellipsoids,pred,"mean")
            std_mat = get_ellipsoid_data(self._ellipsoids,pred,"std")
            zero_dims = zero_std(mat, std_mat, eps_p).item()
            in_ell = is_in_ellipsoid(mat, mean_mat, std_mat, delta_p).item()
            reject_zeros = (zeros_lb > zero_dims) or ((zeros_ub < zero_dims) and z_upper_bound)
            reject_dims = (in_ell_lb > in_ell) or (in_ell_ub < in_ell)
            results.append((reject_zeros or reject_dims, (d == "out dist")))
        
        good_defence = 0
        wrongly_rejected = 0
        num_out_dist = 0
        for rej,out in results:
            if out:
                good_defence += int(rej)
                num_out_dist += 1
            else:
                wrongly_rejected += int(rej)
        print(f"Percentage of good defences: {good_defence/num_out_dist}.")
        print(f"Percentage of wrong rejections: {wrongly_rejected/(len(results)-num_out_dist)}.")

    def get_attacks(self) -> str: return self._adv

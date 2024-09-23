from detection_method.nn import NN
from detection_method.experiments.adversarial_attacks import Experiment


if __name__ == "__main__":
    mlp = NN(
            dataset="mnist",
            path="data/",
            train=True,
            build_matrices=True,
            plot=True,
            epochs=10,
            hidden_size=500,
            num_layers=5
        )
    print(mlp.get_path())
    exp = Experiment(mlp)
    exp.reject_predicted_attacks(std_z=1, eps=0.1, eps_p=0.1)

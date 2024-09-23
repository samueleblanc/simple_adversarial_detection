# Hidden Activations Are Not Enough:
## A General Approach to Neural Network Predictions

This repository contains a very simple version of the code used to run the experiments in the paper titled *Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions*, by Samuel Leblanc, Aiky Rasolomanana, and Marco Armenta. 
The paper can be found on <a href="https://arxiv.org/abs/2409.13163" target="_blank">arXiv</a>. 
The repository for the code used in the paper can be found <a href="https://github.com/MarcoArmenta/Hidden-Activations-are-not-Enough" target="_blank">here</a>. 

**Important:** This code was *not* used to run the experiments of the aforementioned paper. 
To reproduce the experiments, please refer to the repository mentioned above. 

### How to Use

The code being very simple, the file `example.py` contains everything one needs to understand to be able to run an experiment. 

### Out-Of-Distribution Data

The algorithm mentioned in the *Discussion* section of the paper for the detection of out-of-distribution data is implemented. 
To run it, you could for instance add the line `exp.reject_predicted_out_of_dist(std_z=1, std_e=1.5, eps=0.1, eps_p=0.1, delta=2, delta_p=2)` at the end of the file `example.py`. 

### Notation

The variable *t^ε* of the paper refers to `std_z` and *ε =* `eps`, *ε' =* `eps_p`. 
Refering to the *Discussion* section, *t^δ =* `std_e`, *δ =* `delta`, and *δ' =* `delta_p`. 

### CIFAR-10

By changing `dataset="mnist"` to `dataset="cifar10"` in `example.py`, you can train an MLP on CIFAR-10, produce the matrices, and detect adversarial examples!

### License

MIT
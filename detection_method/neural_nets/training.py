import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from detection_method.neural_nets.model import MLP


def get_architecture(input_shape:tuple[int,int,int]=(1, 28, 28), num_classes:int=10, hidden_size:int=500, num_layers:int=5) -> MLP:
    model = MLP(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_sizes=(hidden_size for _ in range(num_layers)),
        bias=True
    )
    return model

def get_dataset(data_set: str, val_set:bool=True) -> tuple:
    # Load the dataset and apply transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if data_set == 'mnist':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    elif data_set == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    elif data_set == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    else:
        print(f"Dataset {data_set} not supported. Only 'mnist', 'fashion', and 'cifar10' are.")
        exit(1)

    if val_set:
        # Split the dataset into train, validation, and test sets
        train_size = int(0.8 * len(train_set))
        val_size = len(train_set) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_set, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        return train_dataset, val_dataset, test_set
    else:
        return train_set, test_set

def train_nn(dataset: str, path: str, lr: float, bs: int, epochs: int, hidden_size:int=500, num_layers:int=5) -> None:
    DEVICE = "cpu"
    SAVE_EVERY = 5
    NUM_CLASSES = 10
    OPTIM = "sgd"
    SAVE_TO = path

    if not os.path.exists(SAVE_TO):
        os.makedirs(f"{SAVE_TO}weights/")
    elif os.path.exists(f"{SAVE_TO}training_results.json"):
        print(f"Experiment at '{SAVE_TO}' already exists.")
        print("Will overwrite.")

    os.makedirs(path, exist_ok=True)

    if dataset == 'cifar10':
        input_shape = (3, 32, 32)
    else:
        input_shape = (1, 28, 28)

    print('Getting network...')
    model = get_architecture(input_shape, NUM_CLASSES, hidden_size, num_layers)

    print('Getting dataset')
    train_dataset, val_dataset, test_dataset = get_dataset(dataset)

    print('data loaders...', flush=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # Initialize the model and optimizer
    print('optimizer...', flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0, lr=lr)

    # Lists to store metrics
    train_losses = []
    val_losses = []
    test_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=78, gamma=0.1)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch: {epoch}", flush=True)
        if epoch == 0:
            torch.save(model.state_dict(), f"{SAVE_TO}weights/epoch{epoch}.pth")

        # Calculate other metrics
        with torch.no_grad():
            val_loss = 0
            total_val = 0
            correct_val = 0
            print("Number of validation batches: ", len(val_loader), flush=True)
            for i, data in enumerate(val_loader, 0):
                if i % 100 == 0:
                    print(f'Valiation batch {i}', flush=True)
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate test accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            test_loss = 0
            total_test = 0
            correct_test = 0
            print("Number of test batches: ", len(val_loader), flush=True)
            for i, data in enumerate(test_loader, 0):
                if i % 100 == 0:
                    print(f'Test batch {i}', flush=True)
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs).to(DEVICE)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Calculate test accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        train_loss = 0.0
        correct_train = 0
        total_train = 0
        print("Number of train batches: ", len(val_loader), flush=True)
        for i, data in enumerate(train_loader, 0):
            if i % 100 == 0:
                print(f'Train batch {i}', flush=True)
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs).to(DEVICE)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        test_loss = test_loss / len(test_loader)

        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val
        test_accuracy = 100 * correct_test / total_test

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if (epoch+1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"{SAVE_TO}weights/epoch{epoch+1}.pth")

        print(f"Epoch {epoch+1}, Training Accuracy: {train_accuracy:.4f}%, Validation Accuracy: {val_accuracy:.4f}%, Test Accuracy: {test_accuracy:.4f}%")
        print(f"         Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f'         Current learning rate: {current_lr}')

    print("Training finished")

    results = {'train_acc': train_accuracies,
            'test_acc': test_accuracies,
            'val_acc': val_accuracies,
            'train_loss': train_losses,
            'test_loss': test_losses,
            'val_loss': val_losses,
            'lr': lr,
            'batch_size': bs,
            'data_set': dataset,
            'architecture': "MLP",
            'type': OPTIM}

    # Save the trained model
    with open(f"{SAVE_TO}training_results.json", "w") as json_file:
        json.dump(results, json_file)

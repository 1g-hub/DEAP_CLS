import os
import shutil
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch
import catalyst
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback
from model import ConvNet
import json


def get_criterion():
    return nn.CrossEntropyLoss()


def get_loaders(batch_size: int = 16, num_workers: int = 4, is_mini: bool = False):
    class MiniMNISTDataset(datasets.MNIST):
        def __init__(self, num_data: int, root: str, train: bool = True, **args: dict):
            super(MiniMNISTDataset, self).__init__(root, train, **args)
            self.data = self.data[:num_data]
            self.targets = self.targets[:num_data]

    transform = transforms.Compose([transforms.ToTensor()])

    # Dataset
    args_dataset = dict(root='./data', download=True, transform=transform)
    trainset = MiniMNISTDataset(num_data=1000, train=True, **args_dataset) if is_mini \
        else datasets.MNIST(train=True, **args_dataset)
    testset = MiniMNISTDataset(num_data=1000, train=False, **args_dataset) if is_mini \
        else datasets.MNIST(train=False, **args_dataset)

    # Data Loader
    args_loader = dict(batch_size=batch_size, num_workers=num_workers)
    train_loader = DataLoader(trainset, shuffle=True, **args_loader)
    val_loader = DataLoader(testset, shuffle=False, **args_loader)
    return train_loader, val_loader


def get_model(num_class: int = 10):
    model = ConvNet(num_class=num_class)
    return model


def get_optimizer(model: torch.nn.Module, init_lr: float = 1e-3, epoch: int = 10):
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(epoch*0.8), int(epoch*0.9)],
        gamma=0.1
    )
    return optimizer, lr_scheduler


def main():
    epochs = 10
    num_class = 10
    output_path = './output/train_mnist/'
    model = get_model(10)
    train_loader, val_loader = get_loaders(is_mini=True)
    loaders = {"train": train_loader, "valid": val_loader}

    optimizer, lr_scheduler = get_optimizer(model=model)
    criterion = get_criterion()

    runner = SupervisedRunner(device=catalyst.utils.get_device())
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loaders=loaders,
        logdir=output_path,
        callbacks=[AccuracyCallback(num_classes=num_class, accuracy_args=[1])],
        num_epochs=epochs,
        main_metric="accuracy01",
        minimize_metric=False,
        fp16=None,
        verbose=True
    )

    is_remove_events = True
    if is_remove_events:
        shutil.rmtree(os.path.join(output_path, "_base_log"))
        shutil.rmtree(os.path.join(output_path, "train_log"))
        shutil.rmtree(os.path.join(output_path, "valid_log"))
        os.remove(os.path.join(output_path, "log.txt"))

    with open(os.path.join(output_path, "checkpoints", "_metrics.json"), encoding="utf-8") as f:
        dic = json.load(f)

    return dic["best"]["accuracy01"]


if __name__ == "__main__":
    main()

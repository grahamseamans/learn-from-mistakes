"""Compare ResNet18 with and without learning from mistakes on CIFAR-10."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np
from collections import deque
import random
from torch.cuda.amp import autocast, GradScaler


class MemoryDataset(Dataset):
    def __init__(self, max_size=5000):
        self.memory = deque(maxlen=max_size)

    def add_mistakes(self, inputs, targets, indices):
        # Detach and store CPU tensors
        for x, y in zip(inputs[indices].detach(), targets[indices].detach()):
            self.memory.append((x.cpu(), y.cpu()))

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        x, y = self.memory[idx]
        # Return tensors that require gradients
        return x.requires_grad_(True), y


def get_scheduler(optimizer, num_epochs, steps_per_epoch):
    def warmup_cosine(step):
        warmup_steps = steps_per_epoch  # 1 epoch of warmup
        total_steps = num_epochs * steps_per_epoch
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
    return scheduler


def train_standard(model, train_loader, test_loader, epochs=10, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.05)
    scheduler = get_scheduler(optimizer, epochs, len(train_loader))
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Standard Epoch {epoch}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Use mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Test accuracy
        test_acc = evaluate(model, test_loader, device)

        print(f"\nStandard Epoch {epoch}:")
        print(f"Train Loss: {train_loss/len(train_loader):.3f}")
        print(f"Train Accuracy: {100.*correct/total:.1f}%")
        print(f"Test Accuracy: {test_acc:.1f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")


def train_with_memory(model, train_loader, test_loader, epochs=10, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.05)
    scheduler = get_scheduler(optimizer, epochs, len(train_loader))
    scaler = GradScaler()
    memory = MemoryDataset(max_size=5000)
    memory_loader = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Memory Epoch {epoch}"):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad_(True)  # Ensure inputs require gradients

            # Regular training step first
            optimizer.zero_grad()

            # Use mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Check for mistakes
            with torch.no_grad():
                _, predicted = outputs.max(1)
                mistakes = ~predicted.eq(targets)

                # If we made mistakes, add them to memory
                if mistakes.any():
                    memory.add_mistakes(inputs, targets, mistakes)

                    # Train on a batch of memories if we have enough
                    if len(memory) >= 64:  # Memory batch size
                        if memory_loader is None or len(memory_loader.dataset) != len(
                            memory
                        ):
                            memory_loader = DataLoader(
                                memory, batch_size=64, shuffle=True
                            )

                        # Sample a batch from memory
                        mem_inputs, mem_targets = next(iter(memory_loader))
                        mem_inputs = mem_inputs.to(device)
                        mem_targets = mem_targets.to(device)

                        # Train on memories with mixed precision
                        optimizer.zero_grad()
                        with autocast():
                            mem_outputs = model(mem_inputs)
                            mem_loss = criterion(mem_outputs, mem_targets)

                        scaler.scale(mem_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

            train_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Test accuracy
        test_acc = evaluate(model, test_loader, device)

        print(f"\nMemory Epoch {epoch}:")
        print(f"Train Loss: {train_loss/len(train_loader):.3f}")
        print(f"Train Accuracy: {100.*correct/total:.1f}%")
        print(f"Test Accuracy: {test_acc:.1f}%")
        print(f"Memory Size: {len(memory)}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(), autocast():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

    # Load CIFAR-10 with standard augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = datasets.CIFAR10(
        "data", train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        "data", train=False, download=True, transform=transform_test
    )

    # Larger batch sizes and more workers
    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
    )

    # Create models
    print("Training standard ResNet18...")
    model_standard = models.resnet18(weights=None, num_classes=10).to(device)
    train_standard(model_standard, train_loader, test_loader)

    print("\nTraining ResNet18 with memory...")
    model_memory = models.resnet18(weights=None, num_classes=10).to(device)
    train_with_memory(model_memory, train_loader, test_loader)

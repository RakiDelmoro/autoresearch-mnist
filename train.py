"""
Autoresearch MNIST training script. Single-GPU, single-file.
Autoresearch context: 2048 tokens, 8192 vocab, 300s time budget.
Dataloader yields (image_tensor, label) tuples for fixed-length batches.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time

import torch
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, make_dataloader, make_mnist_dataloader, download_mnist

# ---------------------------------------------------------------------------
# Model: MLP for MNIST (images are 28x28 pixel tensors, no tokenizer needed)
# ---------------------------------------------------------------------------

@dataclass
class MLPConfig:
    """Model configuration for MNIST classification."""
    image_size: int = 28            # MNIST images are 28x28
    image_channels: int = 1         # grayscale
    hidden_dim: int = 768           # hidden layer dimension (even wider) (wider)
    num_classes: int = 10           # MNIST has 10 classes
    dropout: float = 0.05           # dropout rate
    num_hidden_layers: int = 1      # number of hidden layers (simpler)


class MLP(nn.Module):
    """Simple MLP for MNIST image classification."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        input_dim = config.image_size ** 2 * config.image_channels
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)  # flatten spatial dims after batch
        # Build hidden layers dynamically
        # Build network with proper dimension chaining
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, config.hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(config.dropout))
        # Hidden layers
        current_dim = config.hidden_dim
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(current_dim, config.hidden_dim // 2))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            current_dim = config.hidden_dim // 2
        # Output layer
        layers.append(nn.Linear(current_dim, config.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, images):
        """images: (batch_size, image_size, image_size) float tensor."""
        x = self.flatten(images)  # (batch_size, image_size^2)
        x = self.net(x)
        return x

    def estimate_params(self):
        """Return number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def estimate_flops(self, batch_size):
        """Return estimated FLOPs per batch (forward + backward)."""
        input_dim = self.config.image_size ** 2 * self.config.image_channels
        hidden_dim = self.config.hidden_dim
        hidden_dim2 = self.config.hidden_dim // 2
        num_classes = self.config.num_classes
        num_hidden_layers = self.config.num_hidden_layers

        # Layer 1: (batch, input_dim) -> (batch, hidden_dim)
        flops_1 = 2 * input_dim * hidden_dim
        # Hidden layers: num_hidden_layers layers from hidden_dim to hidden_dim // 2
        flops_hidden = num_hidden_layers * (2 * hidden_dim * hidden_dim2)
        # Output layer: (batch, hidden_dim // 2) -> (batch, num_classes)
        flops_3 = 2 * hidden_dim2 * num_classes

        return flops_1 + flops_hidden + flops_3


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model: MLP, train_loader, val_loader, epochs: int = 1):
    """Train the model on MNIST images.

    Args:
        model: MLP model to train
        train_loader: dataloader yielding (images, labels) tuples
        val_loader: dataloader for validation
        epochs: number of epochs to train

    Returns:
        (model, train_history, val_history)
    """
    device = torch.device("cuda")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            pred = logits.argmax(dim=1)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            running_correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / total

        # Validation
        model.eval()
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # LR schedule
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

    return model, history


def evaluate(model: MLP, dataloader, device):
    """Evaluate model on dataloader. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="mean")
            pred = logits.argmax(dim=1)
            running_loss += loss.item()
            running_correct += (pred == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(dataloader), running_correct / total


# ---------------------------------------------------------------------------
# Main: autoresearch training loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== MNIST Autoresearch Training ===\n")

    # Setup
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    # Build model
    # Note: default num_hidden_layers=2 (can be increased for deeper networks)
    config = MLPConfig()
    model = MLP(config)
    model = torch.compile(model, dynamic=False)
    model.to(device)
    num_params = model.estimate_params()
    num_flops_per_batch = model.estimate_flops(2048)
    print(f"Model params: {num_params:,}")
    print(f"FLOPs per batch (2048 images): {num_flops_per_batch:e}\n")

    # Build dataloaders (fixed-length batches of 2048 images each)
    train_images, train_labels, val_images, val_labels = download_mnist()
    train_loader = make_mnist_dataloader(train_images, train_labels, 2048, num_workers=1)
    val_loader = make_mnist_dataloader(val_images, val_labels, 2048, num_workers=1)
 
     # Training
    print("Starting training...\n")
    model, history = train(model, train_loader, val_loader, epochs=1)

    # Get final validation metrics from history (evaluated at end of each epoch)
    final_val_loss = history['val_loss'][-1]
    final_val_acc = history['val_acc'][-1]

    # Memory usage (captured after training)
    max_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # Summary
    print("\n=== Training Complete ===")
    print(f"Epochs: 1")
    print(f"Final val_loss: {final_val_loss:.4f}")
    print(f"Final val_acc: {final_val_acc:.4f}")
    print(f"Peak memory: {max_memory_gb:.1f} GB")

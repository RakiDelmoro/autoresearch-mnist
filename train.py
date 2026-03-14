"""
Autoresearch MNIST training script. Single-GPU, single-file.
Autoresearch context: 2048 tokens, 8192 vocab, 300s time budget.
Dataloader yields (image_tensor, label) tuples for fixed-length batches.
Now supports both GPU and CPU training.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
import argparse

import numpy as np
import torch
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, make_dataloader, make_mnist_dataloader, download_mnist


def get_device(preferred_device=None):
    """Get the best available device for training.
    
    Args:
        preferred_device: 'cuda', 'cpu', or None (auto-detect)
    
    Returns:
        torch.device object
    """
    if preferred_device == 'cuda':
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
    elif preferred_device == 'cpu':
        return torch.device("cpu")
    else:
        # Auto-detect: prefer CUDA if available
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


def get_memory_stats(device):
    """Get memory statistics for the device.
    
    Args:
        device: torch.device
    
    Returns:
        Memory in GB (0.0 for CPU)
    """
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated() / (1024**3)
    else:
        # CPU memory tracking is complex, return 0.0 for now
        return 0.0

# ---------------------------------------------------------------------------
# Model: MLP for MNIST (images are 28x28 pixel tensors, no tokenizer needed)
# ---------------------------------------------------------------------------

@dataclass
class MLPConfig:
    """Model configuration for MNIST classification."""
    image_size: int = 28            # MNIST images are 28x28
    image_channels: int = 1         # grayscale
    hidden_dim: int = 7168          # amsgrad test with seed=777
    num_classes: int = 10           # MNIST has 10 classes
    dropout: float = 0.1            # back to 0.1
    num_hidden_layers: int = 1      # back to 1 hidden layer


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
        # Input layer -> first hidden
        layers.append(nn.Linear(input_dim, config.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))
        # Additional hidden layers (if any)
        for _ in range(config.num_hidden_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
        # Output layer
        layers.append(nn.Linear(config.hidden_dim, config.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, images):
        """images: (batch_size, image_size, image_size) float tensor."""
        x = self.flatten(images)  # (batch_size, image_size^2)
        x = self.net(x)
        return x

    def estimate_params(self):
        """Return number of parameters."""
        return sum(p.numel() for p in self.net.parameters())

    def estimate_flops(self, batch_size):
        """Return estimated FLOPs per batch (forward pass only)."""
        input_dim = self.config.image_size ** 2 * self.config.image_channels
        hidden_dim = self.config.hidden_dim
        hidden_dim2 = self.config.hidden_dim // 2
        num_classes = self.config.num_classes
        num_hidden_layers = self.config.num_hidden_layers

        # Layer 1: (batch, input_dim) -> (batch, hidden_dim)
        # FLOPs = 2 * input_features * output_features * batch_size (multiply-add ops)
        flops_1 = 2 * input_dim * hidden_dim * batch_size
        # Hidden layers: num_hidden_layers layers from hidden_dim to hidden_dim // 2
        flops_hidden = num_hidden_layers * (2 * hidden_dim * hidden_dim2 * batch_size)
        # Output layer: (batch, hidden_dim // 2) -> (batch, num_classes)
        flops_3 = 2 * hidden_dim2 * num_classes * batch_size

        # Total forward pass FLOPs
        total_flops = flops_1 + flops_hidden + flops_3
        
        return total_flops


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model: MLP, train_loader, val_loader, epochs: int = 1, device=None):
    """Train the model on MNIST images.

    Args:
        model: MLP model to train
        train_loader: dataloader yielding (images, labels) tuples
        val_loader: dataloader for validation
        epochs: number of epochs to train
        device: torch.device to use (if None, defaults to CPU)

    Returns:
        (model, train_history, val_history)
    """
    if device is None:
        device = torch.device("cpu")
    
    model.to(device)
    print(f"Training on device: {device}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # EMA (Exponential Moving Average) for weight averaging
    ema_decay = 0.999
    ema_model = {name: param.clone().detach() for name, param in model.named_parameters()}

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Data augmentation: random shift up to 2 pixels
            if model.training:
                shift_x = torch.randint(-2, 3, (images.size(0),), device=device)
                shift_y = torch.randint(-2, 3, (images.size(0),), device=device)
                for i in range(images.size(0)):
                    images[i] = torch.roll(images[i], shifts=(shift_y[i].item(), shift_x[i].item()), dims=(1, 2))

            # Forward
            logits = model(images)
            # Label smoothing with epsilon=0.1
            epsilon = 0.1
            num_classes = 10
            one_hot = F.one_hot(labels, num_classes).float()
            smooth_labels = one_hot * (1 - epsilon) + epsilon / num_classes
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
            pred = logits.argmax(dim=1)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        ema_model[name] = ema_decay * ema_model[name] + (1 - ema_decay) * param

            # Metrics
            running_loss += loss.item()
            running_correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / total

        # Validation with EMA weights
        # Load EMA weights for validation
        original_weights = {name: param.clone() for name, param in model.named_parameters()}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(ema_model[name])
        
        model.eval()
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        # Restore original weights for next training epoch
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(original_weights[name])

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MNIST Autoresearch Training")
    parser.add_argument('--device', type=str, default=None, 
                        choices=['cuda', 'cpu', 'auto'],
                        help='Device to use for training: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--epochs', type=int, default=11,
                        help='Number of epochs to train (default: 11)')
    args = parser.parse_args()
    
    print("=== MNIST Autoresearch Training ===\n")
    print(f"Device: {args.device if args.device else 'auto'}, Epochs: {args.epochs}")

    # Setup
    device = get_device(args.device)
    torch.manual_seed(777)
    if device.type == 'cuda':
        torch.cuda.manual_seed(777)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()

    # Build model
    config = MLPConfig()
    model = MLP(config)
    
    # Only compile for CUDA (CPU compilation is slower)
    if device.type == 'cuda':
        model = torch.compile(model, dynamic=False)
    
    model.to(device)
    num_params = model.estimate_params()
    num_flops_per_batch = model.estimate_flops(2048)
    print(f"Model params: {num_params:,}")
    print(f"FLOPs per batch (2048 images): {num_flops_per_batch:e}\n")

    # Build dataloaders (fixed-length batches of 2048 images each)
    train_images, train_labels, val_images, val_labels = download_mnist()
    train_loader = make_mnist_dataloader(train_images, train_labels, 512, num_workers=1)
    val_loader = make_mnist_dataloader(val_images, val_labels, 512, num_workers=1)
 
    # Training
    print("Starting training...\n")
    model, history = train(model, train_loader, val_loader, epochs=args.epochs, device=device)

    # Get final validation metrics from history (evaluated at end of each epoch)
    final_val_loss = history['val_loss'][-1]
    final_val_acc = history['val_acc'][-1]

    # Memory usage (captured after training)
    max_memory_gb = get_memory_stats(device)

    # Summary
    print("\n=== Training Complete ===")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Final val_loss: {final_val_loss:.4f}")
    print(f"Final val_acc: {final_val_acc:.4f}")
    if device.type == 'cuda':
        print(f"Peak memory: {max_memory_gb:.1f} GB")
    else:
        print(f"Peak memory: N/A (CPU)")

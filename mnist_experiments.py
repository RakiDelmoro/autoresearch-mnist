"""
MNIST MLP Hyperparameter Experiments
====================================
This script runs a systematic hyperparameter exploration for the MNIST MLP
training task with ~109k parameters.

Experiments:
1. Learning rate effects (1e-4, 5e-4, 1e-3, 3e-4)
2. Weight decay effects (0, 1e-4, 5e-4)
3. Convergence curve (5, 10, 20, 50 epochs)
4. Batch size scaling (64, 128, 256)
5. Learning rate scheduling (none, cosine, step decay)
6. Optimizer comparison (SGD vs AdamW)

Usage:
    python mnist_experiments.py [--dry-run] [--num-epochs PER_EPOCH]

Output:
    - results.tsv: All experiment results
    - mnist_mlp_epoch{epoch}_lr{lr}_wd{wd}_bs{bs}_opt{opt}.pt: Checkpoints
"""

import os
import math
import time
import csv
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prepare import MAX_SEQ_LEN, TIME_BUDGET, make_dataloader, download_mnist, BATCH_SIZE as DEFAULT_BATCH_SIZE

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class MLPConfig:
    """Model configuration for MNIST classification."""
    image_size: int = 28
    image_channels: int = 1
    hidden_dim: int = 128
    num_classes: int = 10
    dropout: float = 0.1


class MLP(nn.Module):
    """Simple MLP for MNIST image classification."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        input_dim = config.image_size ** 2 * config.image_channels
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )

    def forward(self, images):
        x = self.flatten(images)
        x = self.net(x)
        return x

    def estimate_params(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_flops(self, batch_size):
        input_dim = self.config.image_size ** 2 * self.config.image_channels
        hidden_dim = self.config.hidden_dim
        hidden_dim2 = self.config.hidden_dim // 2
        num_classes = self.config.num_classes
        flops_1 = 2 * input_dim * hidden_dim
        flops_2 = 2 * hidden_dim * hidden_dim2
        flops_3 = 2 * hidden_dim2 * num_classes
        return flops_1 + flops_2 + flops_3


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------

def create_optimizer(model, lr, optimizer_name="adamw", weight_decay=0.1):
    """Create optimizer based on name."""
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, scheduler_name="cosine", epochs=None):
    """Create learning rate scheduler."""
    if scheduler_name == "none":
        return None
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs or 1)
    elif scheduler_name == "step10":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def evaluate(model, dataloader, device):
    """Evaluate model on dataloader. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="mean")
            pred = logits.argmax(dim=1)
            running_loss += loss.item()
            running_correct += (pred == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(dataloader), running_correct / total


def train_model(train_loader, val_loader, epochs, device, lr, wd, batch_size,
                optimizer_name, scheduler_name, seed=42):
    """
    Train the model and return history.
    
    Args:
        train_loader: dataloader for training
        val_loader: dataloader for validation
        epochs: number of epochs
        device: torch.device
        lr: learning rate
        wd: weight decay
        batch_size: batch size
        optimizer_name: "adamw" or "sgd"
        scheduler_name: "none", "cosine", or "step10"
        seed: random seed for reproducibility
    
    Returns:
        history dict with train_loss, train_acc, val_loss, val_acc
    """
    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Create model
    model = MLP(MLPConfig())
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, lr, optimizer_name, wd)
    scheduler = create_scheduler(optimizer, scheduler_name, epochs)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    model.to(device)
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            pred = logits.argmax(dim=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / total
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Print progress
        print(f"  Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")
    
    return history


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------


def run_experiment(name, lr, wd, epochs, batch_size, optimizer_name, scheduler_name, device):
    """Run a single experiment and return results."""
    print(f"===========================================================")
    print(f"Experiment: {name}")
    print(f"  lr={lr}, wd={wd}, epochs={epochs}, batch_size={batch_size}, optimizer={optimizer_name}, scheduler={scheduler_name}")
    
    # Create dataloaders (use 0 workers to avoid shared memory issues)
    train_images, train_labels, val_images, val_labels = download_mnist()
    train_loader = make_dataloader(train_images, train_labels, batch_size, num_workers=0)
    val_loader = make_dataloader(val_images, val_labels, batch_size, num_workers=0)
    
    # Start timer
    start_time = time.time()
    
    # Train
    history = train_model(train_loader, val_loader, epochs, device,
                          lr, wd, batch_size, optimizer_name, scheduler_name)
    elapsed_time = time.time() - start_time
    
    # Get final metrics
    final_train_acc = history["train_acc"][-1]
    final_val_acc = history["val_acc"][-1]
    
    # Print results
    print(f"Results:")
    print(f"  Final train_acc: {final_train_acc:.4f}")
    print(f"  Final val_acc: {final_val_acc:.4f}")
    print(f"  Time: {elapsed_time:.1f}s")
    
    # Save results
    results = {
        "name": name,
        "lr": lr,
        "wd": wd,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "train_acc": final_train_acc,
        "val_acc": final_val_acc,
        "time": elapsed_time,
    }
    
   # Save to TSV
    with open("results.tsv", "a") as f:
        f.write(f"{name}\t{lr}\t{wd}\t{epochs}\t{batch_size}\t{optimizer_name}\t{final_train_acc:.4f}\t{final_val_acc:.4f}\t{elapsed_time:.2f}\n")
    
    return results

def run_all_experiments(device):
    """Run all experiments and save results to results.tsv."""
    
    # Define experiments
    experiments = [
        # Convergence curve (batch_size=2048, lr=1e-3, wd=0.1, AdamW, cosine)
        ("Conv_5e",      1e-3, 0.1, 5,    2048, "adamw", "cosine"),
        ("Conv_10e",     1e-3, 0.1, 10,   2048, "adamw", "cosine"),
        ("Conv_20e",     1e-3, 0.1, 20,   2048, "adamw", "cosine"),
        ("Conv_50e",     1e-3, 0.1, 50,   2048, "adamw", "cosine"),
        
        # Learning rate effects (batch_size=2048, wd=0.1, AdamW, cosine)
        ("LR_1e-4",      1e-4, 0.1, 10,   2048, "adamw", "cosine"),
        ("LR_5e-4",      5e-4, 0.1, 10,   2048, "adamw", "cosine"),
        ("LR_1e-3",      1e-3, 0.1, 10,   2048, "adamw", "cosine"),
        ("LR_3e-4",      3e-4, 0.1, 10,   2048, "adamw", "cosine"),
        
        # Weight decay effects (batch_size=2048, lr=1e-3, AdamW, cosine)
        ("WD_0",         1e-3, 0.0, 10,   2048, "adamw", "cosine"),
        ("WD_1e-4",      1e-3, 1e-4, 10,  2048, "adamw", "cosine"),
        ("WD_5e-4",      1e-3, 5e-4, 10,  2048, "adamw", "cosine"),
        
        # Batch size effects (lr=1e-3, wd=0.1, AdamW, cosine)
        ("BS_64",        1e-3, 0.1, 10,   64,   "adamw", "cosine"),
        ("BS_128",       1e-3, 0.1, 10,   128,  "adamw", "cosine"),
        ("BS_256",       1e-3, 0.1, 10,   256,  "adamw", "cosine"),
        
        # Scheduler effects (batch_size=2048, lr=1e-3, wd=0.1, AdamW)
        ("Sch_none",     1e-3, 0.1, 10,   2048, "adamw", "none"),
        ("Sch_cosine",   1e-3, 0.1, 10,   2048, "adamw", "cosine"),
        ("Sch_step10",   1e-3, 0.1, 10,   2048, "adamw", "step10"),
        
        # Optimizer comparison (batch_size=2048, lr=1e-3, wd=0.1, cosine)
        ("Opt_SGD",      1e-3, 0.1, 10,   2048, "sgd",   "cosine"),
        ("Opt_AdamW",    1e-3, 0.1, 10,   2048, "adamw", "cosine"),
    ]
    
    # Run experiments sequentially (to avoid GPU memory issues)
    print(f"\n{'='*60}")
    print(f"Starting all experiments on {device}")
    print(f"Total: {len(experiments)} experiments")
    print(f"{'='*60}\n")
    
    # Track results
    results = []
    
    # Run experiments sequentially
    for exp in experiments:
        result = run_experiment(*exp, device)
        results.append(result)
    
    # Save results to TSV
    output_path = "results.tsv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "lr", "wd", "epochs", "batch_size",
                                              "optimizer", "train_acc", "val_acc", "time"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Name':<12} {'lr':>8} {'wd':>8} {'epochs':>6} {'bs':>5} {'train_acc':>10} {'val_acc':>10} {'time':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<12} {r['lr']:<8.1e} {r['wd']:<8.1e} {r['epochs']:<6} "
              f"{r['batch_size']:<5} {r['train_acc']:<10.4f} {r['val_acc']:<10.4f} "
              f"{r['time']:<8.1f}")
    
    # Sort by val_acc and show best
    sorted_results = sorted(results, key=lambda x: x["val_acc"], reverse=True)
    print(f"\n{'='*60}")
    print("BEST RESULTS (sorted by val_acc)")
    print(f"{'='*60}")
    for r in sorted_results[:5]:
        print(f"  {r['name']:<12}: val_acc = {r['val_acc']:.4f}, time = {r['time']:.1f}s")
    
    return results


def plot_results(results, output_prefix="experiment_plots"):
    """Create plots visualizing the hyperparameter experiment results."""
    # Convert results to structured format for easier plotting
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # 1. Learning rate comparison (all at same epochs, wd, opt settings)
    lr_data = df[df['name'].str.startswith('LR_')].copy()
    lr_data = lr_data.sort_values('lr')
    axes[0].plot(lr_data['lr'], lr_data['val_acc'], 'o-', label='Val Acc', linewidth=2)
    axes[0].plot(lr_data['lr'], lr_data['train_acc'], 's--', label='Train Acc', linewidth=2)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Learning Rate Effects (epochs=10, wd=0.1, AdamW, cosine)')
    axes[0].legend()
    axes[0].grid(True, which='both')
    
    # 2. Weight decay comparison
    wd_data = df[df['name'].str.startswith('WD_')].copy()
    wd_data = wd_data.sort_values('wd')
    axes[1].plot(wd_data['wd'], wd_data['val_acc'], 'o-', label='Val Acc', linewidth=2)
    axes[1].plot(wd_data['wd'], wd_data['train_acc'], 's--', label='Train Acc', linewidth=2)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Weight Decay')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Weight Decay Effects (lr=1e-3, epochs=10, AdamW, cosine)')
    axes[1].legend()
    axes[1].grid(True, which='both')
    
    # 3. Convergence curves (epochs vs accuracy)
    conv_data = df[df['name'].str.startswith('Conv_')].copy()
    epochs_order = [5, 10, 20, 50]
    conv_data = conv_data.set_index('name').loc[[f'Conv_{e}e' for e in epochs_order]].reset_index()
    axes[2].plot(conv_data['epochs'], conv_data['val_acc'], 'o-', label='Val Acc', linewidth=2)
    axes[2].plot(conv_data['epochs'], conv_data['train_acc'], 's--', label='Train Acc', linewidth=2)
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Convergence Curves (lr=1e-3, wd=0.1, AdamW, cosine)')
    axes[2].legend()
    axes[2].grid(True)
    
    # 4. Batch size effects
    bs_data = df[df['name'].str.startswith('BS_')].copy()
    bs_data = bs_data.sort_values('batch_size')
    axes[3].plot(bs_data['batch_size'], bs_data['val_acc'], 'o-', label='Val Acc', linewidth=2)
    axes[3].plot(bs_data['batch_size'], bs_data['train_acc'], 's--', label='Train Acc', linewidth=2)
    axes[3].set_xlabel('Batch Size')
    axes[3].set_ylabel('Accuracy')
    axes[3].set_title('Batch Size Effects (lr=1e-3, wd=0.1, AdamW, cosine)')
    axes[3].legend()
    axes[3].grid(True)
    
    # 5. Scheduler comparison
    sch_data = df[df['name'].str.startswith('Sch_')].copy()
    x = np.arange(len(sch_data))
    width = 0.35
    axes[4].bar(x - width/2, sch_data['val_acc'], width, label='Val Acc')
    axes[4].bar(x + width/2, sch_data['train_acc'], width, label='Train Acc')
    axes[4].set_xticks(x)
    axes[4].set_xticklabels([n.replace('Sch_', '') for n in sch_data['name']])
    axes[4].set_ylabel('Accuracy')
    axes[4].set_title('Scheduler Effects (lr=1e-3, wd=0.1, epochs=10, AdamW)')
    axes[4].legend()
    axes[4].grid(True, axis='y')
    
    # 6. Optimizer comparison
    opt_data = df[df['name'].str.startswith('Opt_')].copy()
    x = np.arange(len(opt_data))
    width = 0.35
    axes[5].bar(x - width/2, opt_data['val_acc'], width, label='Val Acc')
    axes[5].bar(x + width/2, opt_data['train_acc'], width, label='Train Acc')
    axes[5].set_xticks(x)
    axes[5].set_xticklabels([n.replace('Opt_', '') for n in opt_data['name']])
    axes[5].set_ylabel('Accuracy')
    axes[5].set_title('Optimizer Comparison (lr=1e-3, wd=0.1, epochs=10, cosine)')
    axes[5].legend()
    axes[5].grid(True, axis='y')
    
    plt.tight_layout()
    plot_path = f"{output_prefix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to {plot_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST MLP Hyperparameter Experiments")
    parser.add_argument("--dry-run", action="store_true", help="List experiments without running")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override epochs per experiment")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = torch.device("cpu")
    
    if args.dry_run:
        print("\n=== DRY RUN: Experiment List ===\n")
        print(f"{'Name':<12} {'lr':>8} {'wd':>8} {'epochs':>6} {'bs':>5} {'opt':>6} {'sched':>8}")
        print("-" * 60)
        for exp in run_all_experiments(device):
            pass  # Just showing the list
        print(f"\nTotal: {len(run_all_experiments(device))} experiments")
    else:
        # First check if data is downloaded
        if not os.path.exists(os.path.expanduser("~/.cache/autoresearch/mnist")):
            print("Downloading MNIST data...")
            train_images, train_labels, val_images, val_labels = download_mnist()
        
        results = run_all_experiments(device)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        best = max(results, key=lambda x: x["val_acc"])
        print(f"\nBest result: {best['name']:<12} -> val_acc = {best['val_acc']:.4f}")
        print(f"Total time: {sum(r['time'] for r in results):.1f}s")
        
        # Generate plots
        print("\n" + "="*60)
        print("GENERATING PLOTS")
        print("="*60)
        plot_results(results)

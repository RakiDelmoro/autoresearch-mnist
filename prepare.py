"""
Data preparation for autoresearch MNIST experiments.

Downloads MNIST and creates dataloaders.

Usage:
    python prepare.py      # download + setup (full prep)
    python prepare.py --num-workers 1   # use 1 worker (for testing)

Data is stored in ~/.cache/autoresearch/.
"""

# Training constants
MAX_SEQ_LEN = 2048
TIME_BUDGET = 300  # seconds
BATCH_SIZE = 128

import os
import sys
import math
import pickle
import argparse
from multiprocessing import Pool

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
MNIST_DIR = os.path.join(CACHE_DIR, "mnist")
MNIST_URL = "https://www.cs.toronto.edu/~kriz/"



# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_file(path):
    """Download a single MNIST file."""
    try:
        torchvision.datasets.MNIST.download(root=MNIST_DIR)
        return True
    except Exception as e:
        print(f"  Failed to download {path}: {e}")
        return False


def download_mnist():
    """Download MNIST training and validation datasets."""
    os.makedirs(MNIST_DIR, exist_ok=True)

    # Check if already downloaded
    train_path = os.path.join(MNIST_DIR, "train-images-idx3-ubyte")
    val_path = os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte")
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"MNIST: already downloaded at {MNIST_DIR}")
        # Load datasets to extract images and labels
        train_dataset = torchvision.datasets.MNIST(root=MNIST_DIR, train=True, download=False)
        val_dataset = torchvision.datasets.MNIST(root=MNIST_DIR, train=False, download=False)
        train_images = train_dataset.data.unsqueeze(1).float() / 255.0
        train_labels = train_dataset.targets
        val_images = val_dataset.data.unsqueeze(1).float() / 255.0
        val_labels = val_dataset.targets
        return train_images, train_labels, val_images, val_labels

    # Download — use new PyTorch API that handles both files automatically
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root=MNIST_DIR)

    print(f"MNIST: downloaded to {MNIST_DIR}")

    # Load and convert to Float tensors — model expects Float
    print("Converting to tensors...")
    train_dataset = torchvision.datasets.MNIST(root=MNIST_DIR, train=True, download=False)
    val_dataset = torchvision.datasets.MNIST(root=MNIST_DIR, train=False, download=False)

    # Images: [N, 1, 28, 28] float tensors normalized to [0,1]
    train_images = train_dataset.data.unsqueeze(1).float() / 255.0
    train_labels = train_dataset.targets
    val_images = val_dataset.data.unsqueeze(1).float() / 255.0
    val_labels = val_dataset.targets

    print(f"MNIST: {train_images.shape} images, {train_labels.shape[0]} labels")
    print(f"MNIST: {val_images.shape} images, {val_labels.shape[0]} labels")

    return train_images, train_labels, val_images, val_labels


def download_fashion_mnist():
    """Download Fashion-MNIST training and validation datasets."""
    FASHION_DIR = os.path.join(CACHE_DIR, "fashion")
    os.makedirs(FASHION_DIR, exist_ok=True)

    # Check if already downloaded
    train_path = os.path.join(FASHION_DIR, "train")
    val_path = os.path.join(FASHION_DIR, "t10k")
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Fashion-MNIST: already downloaded at {FASHION_DIR}")
        # Load datasets to extract images and labels
        train_dataset = torchvision.datasets.FashionMNIST(root=FASHION_DIR, train=True, download=False)
        val_dataset = torchvision.datasets.FashionMNIST(root=FASHION_DIR, train=False, download=False)
        train_images = train_dataset.data.unsqueeze(1).float() / 255.0
        train_labels = train_dataset.targets
        val_images = val_dataset.data.unsqueeze(1).float() / 255.0
        val_labels = val_dataset.targets
        return train_images, train_labels, val_images, val_labels

    # Download — use new PyTorch API that handles both files automatically
    print("Downloading Fashion-MNIST...")
    torchvision.datasets.FashionMNIST(root=FASHION_DIR)

    print(f"Fashion-MNIST: downloaded to {FASHION_DIR}")

    # Load and convert to Float tensors — model expects Float
    print("Converting to tensors...")
    train_dataset = torchvision.datasets.FashionMNIST(root=FASHION_DIR, train=True, download=False)
    val_dataset = torchvision.datasets.FashionMNIST(root=FASHION_DIR, train=False, download=False)

    # Images: [N, 1, 28, 28] float tensors normalized to [0,1]
    train_images = train_dataset.data.unsqueeze(1).float() / 255.0
    train_labels = train_dataset.targets
    val_images = val_dataset.data.unsqueeze(1).float() / 255.0
    val_labels = val_dataset.targets

    print(f"Fashion-MNIST: {train_images.shape} images, {train_labels.shape[0]} labels")
    print(f"Fashion-MNIST: {val_images.shape} images, {val_labels.shape[0]} labels")

    return train_images, train_labels, val_images, val_labels


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def make_mnist_dataloader(images, labels, batch_size, num_workers=8, split="train"):
    """
    Create a dataloader for MNIST data.

    Args:
        images: Tensor of shape [N, 1, H, W]
        labels: Tensor of shape [N]
        batch_size: Batch size per worker
        num_workers: Number of parallel workers
        split: "train" or "val"

    Returns:
        Dataloader yielding (image_tensor, label) tuples
    """
    dataset = torch.utils.data.TensorDataset(images, labels)
    sampler = torch.utils.data.SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                     num_workers=num_workers, pin_memory=True)


def make_fashion_dataloader(images, labels, batch_size, num_workers=8, split="train"):
    """
    Create a dataloader for Fashion-MNIST data.

    Args:
        images: Tensor of shape [N, 1, H, W]
        labels: Tensor of shape [N]
        batch_size: Batch size per worker
        num_workers: Number of parallel workers
        split: "train" or "val"

    Returns:
        Dataloader yielding (image_tensor, label) tuples
    """
    dataset = torch.utils.data.TensorDataset(images, labels)
    sampler = torch.utils.data.SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                     num_workers=num_workers, pin_memory=True)


def make_dataloader(images, labels, batch_size, num_workers=8):
    """
    Autoresearch dataloader interface for MNIST.

    Yields (image_tensor, label) for each batch.

    Args:
        images: Tensor of shape [N, 1, H, W]
        labels: Tensor of shape [N]
        batch_size: Batch size per worker
        num_workers: Number of parallel workers

    Returns:
        Dataloader yielding (image_tensor, label) tuples
    """
    return make_mnist_dataloader(images, labels, batch_size, num_workers)


def make_fashion(images, labels, batch_size, num_workers=8):
    """
    Autoresearch dataloader interface for Fashion-MNIST.

    Yields (image_tensor, label) for each batch.

    Args:
        images: Tensor of shape [N, 1, H, W]
        labels: Tensor of shape [N]
        batch_size: Batch size per worker
        num_workers: Number of parallel workers

    Returns:
        Dataloader yielding (image_tensor, label) tuples
    """
    return make_fashion_dataloader(images, labels, batch_size, num_workers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MNIST data for autoresearch")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for dataloaders")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download MNIST
    train_images, train_labels, val_images, val_labels = download_mnist()
    print()

    # Step 2: Create dataloaders
    train_loader = make_dataloader(train_images, train_labels, BATCH_SIZE, args.num_workers)
    val_loader = make_dataloader(val_images, val_labels, BATCH_SIZE, args.num_workers)

    print(f"Train dataloader: {train_loader.batch_size} samples/worker")
    print(f"Val dataloader: {val_loader.batch_size} samples/worker")
    print()
    print("Done! Ready to train.")

# autoresearch-mnist
AI agents running research on MNIST task

## Setup
Branch: `autoresearch/mar13` | Data: `~/.cache/autoresearch/mnist/`

## Accomplished
- Created branch `autoresearch/mar13`
- Downloaded MNIST training (100k images) and validation (10k images) sets
- Converted datasets to PyTorch tensors
- Fixed bug in `prepare.py`: `MNIST.download()` was being called with incorrect `root` parameter
- Removed unused autoresearch config variables from `prepare.py` (left only MNIST-specific params)
- `results.tsv` ready for logging

## Training
See `train.py` — MLP model with `torch.compile` enabled, `dynamic=False`

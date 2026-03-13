# autoresearch-mnist
AI agents running research on MNIST image classification

## Quick Start

Follow the experiment protocol in `program.md`. This file contains the complete instructions for autonomous research, including the experiment loop, logging, and success criteria.

**In short:** Only modify `train.py`. Run experiments, log results to `results.tsv`, keep improving `val_acc`, never stop.

## Experiment Branch
`autoresearch/mar13-v2`

## Best Result (as of Mar 13, 2026)
**Validation Accuracy: 0.9237** (92.37%)

**Best Configuration:**
- Architecture: Single hidden layer MLP
- Hidden dimension: 4096
- Dropout: 0.05
- Optimizer: AdamW (lr=0.001, weight_decay=0.1)
- Batch size: 2048
- Epochs: 1
- Peak VRAM: 0.3 GB

## Data
Location: `~/.cache/autoresearch/mnist/`

## Running Experiments
```bash
python train.py > run.log 2>&1
```

Results are logged to `results.tsv`. See `program.md` for full protocol.

## Files
- `train.py` - Training script (only file to modify)
- `prepare.py` - Data preparation (fixed)
- `results.tsv` - Experiment results
- `plot_experiments.py` - Visualization script


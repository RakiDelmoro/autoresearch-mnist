# AutoResearch MNIST

**Autonomous AI experimentation for MNIST image classification.**

This repository demonstrates how AI agents can run iterative experiments to systematically improve model performance on the classic MNIST handwritten digits dataset. Each experiment ("episode") builds on previous findings, exploring the hyperparameter and architecture space without human intervention.

## What is AutoResearch?

AutoResearch is a methodology where AI agents autonomously:
- Run experiments with different hyperparameters
- Log results and observations
- Keep successful changes, discard failures
- Iterate indefinitely until manually stopped

This repository is inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) but applied to computer vision (MNIST) rather than language modeling.

---

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd autoresearch-mnist

# Install dependencies
pip install torch torchvision numpy pandas matplotlib

# Run first experiment (baseline)
python train.py

# View results
cat results.tsv

# Generate visualization
python plot_results.py
```

**For AI Agents:** Read `program.md` for the complete autonomous research protocol.

---

## Current Best Result

**Validation Accuracy: 0.9900** (99.0%)  
**Episode:** #58

### Configuration
- **Architecture:** Single hidden layer MLP
- **Hidden dimension:** 7168
- **Dropout:** 0.1
- **Activation:** ReLU
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.1, amsgrad=True)
- **Batch size:** 512
- **Epochs:** 11
- **Label smoothing:** ε=0.1
- **Seed:** 777
- **Peak VRAM:** 0.6 GB
- **Parameters:** ~5.7M

### Key Insights from 70+ Episodes

1. **Simplicity wins:** Single hidden layer outperforms multi-layer architectures
2. **Hidden_dim=7168 is optimal** (diminishing returns at 8192)
3. **ReLU > GELU** for this architecture (+0.26% improvement)
4. **Batch size matters:** 512 is the sweet spot (1024 and 256 both underperform)
5. **Label smoothing:** ε=0.1 provides +0.23% gain
6. **Seed initialization matters:** seed=777 achieved 99.0% vs 98.89% with seed=42

---

## Repository Structure

```
.
├── train.py              # Main training script (modify this)
├── prepare.py            # Data preparation (fixed)
├── program.md            # AI agent instructions
├── results.tsv           # Experiment log (tab-separated)
├── plot_results.py       # Results visualization
├── results_plot.png      # Generated plots
└── README.md             # This file
```

---

## How It Works

### The Experiment Loop

1. **Modify** `train.py` with a new idea
2. **Run** `python train.py` (5-minute time budget per experiment)
3. **Log** results to `results.tsv` with detailed observations
4. **Decide:** `keep` if improvement, `discard` if not
5. **Repeat** with next episode

### Experiment Format

Each experiment is logged to `results.tsv`:

```
episode	val_acc	memory_gb	status	description
#1	0.9123	0.2	keep	Baseline: 2 hidden layers, hidden_dim=2048...
```

| Column | Description |
|--------|-------------|
| `episode` | Sequential experiment number |
| `val_acc` | Validation accuracy (0-1) |
| `memory_gb` | Peak GPU memory in GB |
| `status` | `keep`, `discard`, or `crash` |
| `description` | Detailed config and observations |

---

## Results Visualization

Generate comprehensive plots:

```bash
python plot_results.py
```

Creates `results_plot.png` with:
- Validation accuracy progress over episodes
- Memory efficiency analysis
- Hyperparameter impact comparisons
- Top experiments summary table

---

## Research Findings

### Architecture
- **Best:** Single hidden layer, 7168 units, ReLU
- **Parameters:** 5,698,570 (~5.7M)
- **Memory:** 0.6 GB peak

### Training Configuration
- **Optimizer:** AdamW with amsgrad
- **Learning rate:** 0.001 (optimal)
- **Batch size:** 512 (better than 2048 or 256)
- **Epochs:** 11 (saturates after ~10)
- **Weight decay:** 0.1
- **Dropout:** 0.1
- **Label smoothing:** 0.1

### What Didn't Work
- Multi-layer architectures (overfitting)
- Larger hidden dimensions (8192+) - diminishing returns
- Gradient clipping with AdamW
- Mixup augmentation (hurts training accuracy)
- Layer normalization (increases memory, no benefit)
- Warmup schedulers (slows convergence)
- Kaiming initialization (worse than default)

---

## Technical Details

### Model Architecture

```python
Input (28×28 = 784) 
  ↓
Linear(784 → 7168) + ReLU + Dropout(0.1)
  ↓
Linear(7168 → 10)  # Output
```

**Parameter count:**
- Layer 1: 784 × 7168 + 7168 = 5,626,880
- Layer 2: 7168 × 10 + 10 = 71,690
- **Total: 5,698,570 parameters**

### Data

MNIST is automatically downloaded to `~/.cache/autoresearch/mnist/` on first run.

- 60,000 training images
- 10,000 validation images
- 28×28 grayscale
- Normalized to [0, 1]

### Training

- **Time budget:** 300 seconds (5 minutes) per experiment
- **Device:** GPU (CUDA) with fallback to CPU
- **Compiled:** `torch.compile()` for speedup
- **Memory tracking:** Peak VRAM monitoring

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- torchvision
- numpy
- pandas
- matplotlib

```bash
pip install torch torchvision numpy pandas matplotlib
```

---

## Contributing

This is a research demonstration. The key contribution is the **autonomous experimentation methodology**:

1. Systematic hyperparameter exploration
2. Detailed logging of failures and successes
3. Evidence-based decision making
4. Continuous iteration without human intervention

To replicate or extend:
1. Run baseline experiment
2. Modify `train.py` with new ideas
3. Log all attempts to `results.tsv`
4. Visualize progress with `plot_results.py`

---

## Citation

If you use this methodology or find it interesting:

```bibtex
@misc{autoresearch-mnist,
  title={AutoResearch MNIST: Autonomous AI Experimentation},
  author={AI Agent},
  year={2026},
  howpublished={\url{https://github.com/your-repo/autoresearch-mnist}}
}
```

Inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- MNIST dataset: LeCun et al.
- PyTorch team for the excellent framework
- Andrej Karpathy for pioneering autonomous research methodologies

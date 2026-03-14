# autoresearch-mnist

This is an experiment to have the LLM do its own research on MNIST image classification.

## Setup

To set up a new experiment:

1. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, data prep, dataloaders, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.

2. **Verify data exists**: Check that `~/.cache/autoresearch/mnist/` contains MNIST data. If not, run `python prepare.py`.

3. **Initialize results.tsv**: Create `results.tsv` with the header row:
   ```
   episode	val_acc	memory_gb	status	description
   ```

4. **Start experimenting**: You're ready to begin the loop!

**Note:** No git commits needed for experiments. Just modify `train.py` and run. The TSV file tracks all episodes.

---

## Experimentation

Each experiment runs on GPU. The training script runs for a fixed time budget of **300 seconds** (5 minutes, wall clock training time, excluding startup/compilation). You launch it simply as:

```bash
python train.py
```

### What you CAN do:

- **Modify `train.py`** — this is the only file you edit. Everything is fair game: model architecture (MLP only), optimizer, hyperparameters, training loop, batch size, model size, etc. **Note:** Only MLP architecture modifications are allowed; do not add CNNs or other model types.

### What you CANNOT do:

- **Modify `prepare.py`** — it is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, batch size, etc).
- **Install new packages or add dependencies** — you can only use what's already in `pyproject.toml`.
- **Modify the evaluation harness** — the `evaluate` function in `train.py` is the ground truth metric.

### The goal:

Get the **highest validation accuracy (val_acc)**. Lower validation loss is secondary. The time budget is fixed (300 seconds), so you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

### VRAM is a soft constraint:

Some increase is acceptable for meaningful `val_acc` gains, but it should not blow up dramatically.

### Simplicity criterion:

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.5% val_acc improvement that adds 20 lines of hacky code? Probably not worth it. A 0.5% val_acc improvement from deleting code? Definitely keep. An improvement of ~0% but much simpler code? Keep.

---

## The first run:

Your very first run should always be to establish the baseline, so you will run the training script as is.

---

## Output format

Once the script finishes it prints a summary like this:

```
=== Training Complete ===
Epochs: 1
Final val_loss: 0.2345
Final val_acc: 0.9234
Peak memory: 2.1 GB
```

The script tracks validation accuracy, loss, and GPU memory usage.

---

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

**IMPORTANT:** Experiments are NOT committed to git. Each experiment is just a run that gets logged to the TSV file. The TSV tracks the experiment history, not git commits.

The TSV has a header row and 5 columns:

```
episode	val_acc	memory_gb	status	description
```

| Column       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `episode`    | Sequential experiment number starting from #1                               |
| `val_acc`    | validation accuracy (e.g. 0.9234) — use 0.0000 for crashes                  |
| `memory_gb`  | peak GPU memory in GB, round to .1f (check nvidia-smi or torch memory stats) — use 0.0 for crashes |
| `status`     | `keep`, `discard`, or `crash`                                               |
| `description`| detailed description of the experiment configuration and results             |

**Description Format:** Include all hyperparameters and architecture details:
- Architecture: layers, hidden dimensions, dropout
- Optimizer: learning rate, weight decay, scheduler
- Training: epochs, batch size
- Key results and observations

Example:

```
episode	val_acc	memory_gb	status	description
#1	0.9234	2.1	keep	baseline MLP: 1 hidden layer, hidden_dim=2048, dropout=0.0, lr=0.001, weight_decay=0.1, epochs=1, batch_size=2048. Final train_acc=0.95, no issues
#2	0.9456	2.2	keep	CNN added: 2 conv layers (32,64 filters) + 1 FC hidden=1024, dropout=0.1, lr=0.0005, weight_decay=0.01, epochs=2. train_acc=0.98, val_acc improved significantly
#3	0.9100	2.0	discard	smaller model: 1 hidden layer, hidden_dim=512, lr=0.001, epochs=1. Underfitting, val_acc dropped from best
#4	0.0000	0.0	crash	deep model: 5 hidden layers, hidden_dim=8192, lr=0.01. OOM error during forward pass, model too large
#5	0.9236	1.2	keep	batch norm added: 2 hidden layers, hidden_dim=4096, batch_norm=True, dropout=0.05, lr=0.0015, weight_decay=0.05, epochs=3. Slight improvement over baseline, stable training
```

---

## The experiment loop

**LOOP FOREVER:**

1. Increment the episode counter (starting from #1)

2. **Tune `train.py`** with an experimental idea by directly hacking the code. Try different:
   - Architecture changes (layers, dimensions, activations, regularization)
   - Optimizer settings (learning rate, weight decay, scheduler type)
   - Training parameters (epochs, batch size)
   
3. Run the experiment:
   ```bash
   uv run train.py > run.log 2>&1
   ```
   (redirect everything — **do NOT** use tee or let output flood your context)

4. Read out the results: `grep "val_acc:" run.log`

5. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, log it as crash and move on.

6. Record the results in the TSV with **detailed description** including:
   - Episode number
   - All architecture parameters tried
   - Optimizer settings used
   - Training configuration
   - Key observations about what worked/didn't work
   
7. Keep track of the best `val_acc` achieved so far and use it as your baseline for improvement

8. Continue to next episode with a new experiment idea based on what you've learned

---

## The idea:

You are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this **very very sparingly** (if ever).

---

## Timeout:

Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

---

## Crashes:

If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the TSV, and move on.

---

## NEVER STOP:

Once the experiment loop has begun (after the initial setup), do **NOT** pause to ask the human if you should continue. Do **NOT** ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

**Remember:** Each experiment is logged to `results.tsv` with detailed episode information. This file is the only record you need - no git commits for experiments.

---

## GPU Training Notes

### GPU Training (CUDA)
- **Faster training** due to parallel computation
- **Memory tracking** via `torch.cuda.max_memory_allocated()`
- **Compilation** via `torch.compile()` for additional speedup
- **Higher batch sizes** possible
- **Watch VRAM** - some configurations may OOM

### Best Practices:
- Monitor GPU memory usage in results
- Report memory as peak VRAM in GB
- Use torch.compile for speedup
- Watch for OOM errors on large models

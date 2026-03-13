# autoresearch-mnist

This is an experiment to have the LLM do its own research on MNIST image classification.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. mar13). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.

2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.

3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloaders, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.

4. **Verify data exists**: Check that `~/.cache/autoresearch/mnist/` contains MNIST data. If not, tell the human to run `python prepare.py`.

5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.

6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

---

## Experimentation

Each experiment runs on a single GPU. The training script runs for a fixed time budget of **300 seconds** (5 minutes, wall clock training time, excluding startup/compilation). You launch it simply as:

```bash
uv run train.py
```

### What you CAN do:

- **Modify `train.py`** — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

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
```

The script tracks validation accuracy and loss.

---

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_acc	memory_gb	status	description
```

| Column       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `commit`     | git commit hash (short, 7 chars)                                            |
| `val_acc`    | validation accuracy (e.g. 0.9234) — use 0.0000 for crashes                 |
| `memory_gb`  | peak memory in GB, round to .1f (check nvidia-smi or torch memory stats) — use 0.0 for crashes |
| `status`     | `keep`, `discard`, or `crash`                                              |
| `description`| short text description of what this experiment tried                        |

Example:

```
commit	val_acc	memory_gb	status	description
a1b2c3d	0.9234	2.1	keep	baseline MLP
b2c3d4e	0.9456	2.2	keep	add CNN layers
c3d4e5f	0.9100	2.0	discard	switch to tiny model
d4e5f6g	0.0000	0.0	crash	too many filters (OOM)
```

---

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar13` or `autoresearch/mar13-gpu0`).

**LOOP FOREVER:**

1. Look at the git state: the current branch/commit we're on

2. **Tune `train.py`** with an experimental idea by directly hacking the code.

3. `git commit`

4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — **do NOT** use tee or let output flood your context)

5. Read out the results: `grep "val_acc:" run.log`

6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.

7. Record the results in the TSV (NOTE: **do not commit the results.tsv file, leave it untracked by git**)

8. If `val_acc` improved (higher), you "advance" the branch, keeping the git commit

9. If `val_acc` is equal or worse, you `git reset` back to where you started

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

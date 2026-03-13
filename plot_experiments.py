#!/usr/bin/env python3
"""Plot experiment results from results.tsv."""
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results.tsv', sep='\t')

# Filter out any crashes (val_acc=0)
df = df[df['val_acc'] > 0]

# Convert commit to sequential experiment number (preserve order from file)
df['exp_num'] = range(1, len(df) + 1)

# Identify best result
best_idx = df['val_acc'].idxmax()
best_exp = df.loc[best_idx]

# Plot
plt.figure(figsize=(12, 6))
colors = ['green' if status == 'keep' else 'red' for status in df['status']]
plt.scatter(df['exp_num'], df['val_acc'], c=colors, s=100, alpha=0.7, edgecolors='k')
plt.plot(df['exp_num'], df['val_acc'], 'k--', alpha=0.5, linewidth=1)
plt.xlabel('Experiment Number (chronological)')
plt.ylabel('Validation Accuracy')
plt.title('MNIST Autoresearch Experiment Progress')
plt.grid(True, alpha=0.3)

# Annotate best
plt.scatter(best_exp['exp_num'], best_exp['val_acc'], c='gold', s=200, marker='*', edgecolors='k', zorder=5)
plt.annotate(f"Best: {best_exp['val_acc']:.4f}\n{best_exp['description']}",
             xy=(best_exp['exp_num'], best_exp['val_acc']),
             xytext=(best_exp['exp_num']+0.5, best_exp['val_acc']-0.02),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9)

plt.tight_layout()
plt.savefig('experiment_progress.png', dpi=150)
print(f"Plot saved to experiment_progress.png")
print(f"Best experiment: #{best_exp['exp_num']} - {best_exp['description']} (val_acc={best_exp['val_acc']:.4f})")
plt.show()

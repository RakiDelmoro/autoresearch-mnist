import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read results
df = pd.read_csv('results.tsv', sep='\t')

# Set style
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Accuracy by configuration
sns.barplot(data=df, x='config', y='accuracy', ax=axes[0, 0], errorbar=None, alpha=0.7)
axes[0, 0].set_title('Best Accuracy by Configuration')
axes[0, 0].set_xlabel('Configuration')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Accuracy vs Time (efficiency)
sns.scatterplot(data=df, x='time_s', y='accuracy', hue='config', ax=axes[0, 1], alpha=0.6, s=60)
axes[0, 1].set_title('Accuracy vs Training Time (lower is better)')
axes[0, 1].set_xlabel('Time (seconds)')
axes[0, 1].set_ylabel('Accuracy')

# 3. Accuracy by Hyperparameter
for param in ['lr', 'batch_size', 'epochs']:
    sns.boxplot(data=df, x=param, y='accuracy', ax=axes[1, 0])
axes[1, 0].set_title(f'Accuracy by {param}')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Accuracy by Weight Decay
sns.boxplot(data=df, x='weight_decay', y='accuracy', ax=axes[1, 1])
axes[1, 1].set_title('Accuracy by Weight Decay')
axes[1, 1].tick_params(axis='x', rotation=0)

# Add best performer label
best = df.loc[df['accuracy'].idxmax()]
axes[0, 0].text(best['config'] + 0.02, best['accuracy'] + 0.01, f"Best: {best['accuracy']:.3f}",
                fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('results_summary.png', dpi=150)
print("Visualization saved to results_summary.png")

# Print top performers
print("\n" + "="*60)
print("TOP 10 BEST PERFORMERS")
print("="*60)
print(df.sort_values('accuracy', ascending=False).head(10)[['config', 'accuracy', 'epoch', 'lr', 'batch_size', 'weight_decay', 'time_s']])

print("\n" + "="*60)
print("BEST CONFIGURATION")
print("="*60)
best = df.loc[df['accuracy'].idxmax()]
for col in ['config', 'accuracy', 'epoch', 'lr', 'batch_size', 'weight_decay', 'time_s', 'device']:
    print(f"  {col}: {best[col]}")

print("\n" + "="*60)
print("RECOMMENDED CONFIGURATION FOR FINAL TRAINING")
print("="*60)
best = df.loc[df['accuracy'].idxmax()]
print(f"""
python train.py --epochs {int(best['epoch'])} --lr {best['lr']} --batch-size {int(best['batch_size'])} --weight-decay {best['weight_decay']} --device cuda:0
""")

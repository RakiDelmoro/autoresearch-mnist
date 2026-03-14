#!/usr/bin/env python3
"""Plot results from results.tsv - enhanced visualization"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re


def parse_episode_line(line):
    """Parse an episode line and extract key fields from description."""
    line = line.strip()
    # Remove line number prefix
    line = re.sub(r'^\d+:\s*', '', line)
    
    # Basic split
    parts = line.split(None, 4)
    if len(parts) < 5:
        return None
    
    episode, val_acc, memory_gb, status, description = parts
    
    # Parse description for key fields
    desc_lower = description.lower()
    
    # Extract architecture
    arch = 'Unknown'
    if 'single hidden layer' in desc_lower:
        arch = 'Single hidden layer'
    elif 'deeper' in desc_lower or '2 layers' in desc_lower:
        arch = '2 hidden layers'
    elif 'deeper constant-width' in desc_lower:
        arch = 'Deeper constant-width'
    
    # Extract hidden_dim
    hidden_dim_match = re.search(r'hidden_dim=(\d+)', description)
    hidden_dim = int(hidden_dim_match.group(1)) if hidden_dim_match else None
    
    # Extract activation
    activation = 'GELU'
    if 'relu' in desc_lower:
        activation = 'ReLU'
    elif 'silu' in desc_lower or 'swish' in desc_lower:
        activation = 'SiLU'
    elif 'gelu' in desc_lower and 'relu' not in desc_lower:
        activation = 'GELU'
    
    # Extract epochs
    epochs_match = re.search(r'epochs=(\d+)', description)
    epochs = int(epochs_match.group(1)) if epochs_match else None
    
    # Extract val_loss (stop at space, dot, or comma)
    val_loss_match = re.search(r'val_loss=([\d.]+)(?:\s|$|,|\.)', description)
    # Remove trailing dot if present (can happen with greedy matching)
    val_loss_str = val_loss_match.group(1).rstrip('.') if val_loss_match else None
    val_loss = float(val_loss_str) if val_loss_str else None
    
    # Extract train_acc
    train_acc_match = re.search(r'train_acc=([\d.]+)', description)
    train_acc = float(train_acc_match.group(1)) if train_acc_match else None
    
    # Extract lr
    lr_match = re.search(r'lr=([\d.]+)', description)
    lr = float(lr_match.group(1)) if lr_match else None
    
    # Extract batch_size
    batch_match = re.search(r'batch_size=(\d+)', description)
    batch_size = int(batch_match.group(1)) if batch_match else None
    
    # Extract weight_decay
    wd_match = re.search(r'weight_decay=([\d.]+)', description)
    weight_decay = float(wd_match.group(1)) if wd_match else None
    
    # Extract epsilon (label smoothing)
    eps_match = re.search(r'epsilon=([\d.]+)', description)
    epsilon = float(eps_match.group(1)) if eps_match else None
    
    # Extract params
    params_match = re.search(r'params?[\s:]*(\d+[,M]+)', description)
    params_str = params_match.group(1) if params_match else None
    params = float(params_str.replace(',', '').replace('M', '')) * 1e6 if params_str else None
    
    # Extract overfitting info
    overfit = 'No'
    if 'overfit' in desc_lower or 'slight overfit' in desc_lower:
        overfit = 'Yes'
    
    # Extract peak val_acc
    peak_match = re.search(r'peak\??:\s*([\d.]+)', description)
    peak_val = float(peak_match.group(1)) if peak_match else None
    
    # Extract peak epoch
    peak_ep_match = re.search(r'peak at epoch (\d+)', description)
    peak_epoch = int(peak_ep_match.group(1)) if peak_ep_match else None
    
    # Extract model size
    size_match = re.search(r'(\d+\.?\d*)M params', description)
    model_size = float(size_match.group(1)) if size_match else None
    
    # Extract best val_acc (for comparison)
    best_match = re.search(r'best\s*[:\s]+([\d.]+)', description)
    best_val_str = best_match.group(1).rstrip('.') if best_match else None
    best_val = float(best_val_str) if best_val_str else None
    
    # Extract val_acc progression for kept experiments
    val_progression = []
    val_pattern = re.findall(r'val_acc[=:]?\s*([\d.]+)', description)
    if val_pattern:
        # Clean up trailing dots from extracted values
        val_progression = [float(v.rstrip('.')) for v in val_pattern]
    
    return {
        'episode': episode,
        'val_acc': float(val_acc),
        'memory_gb': float(memory_gb),
        'status': status,
        'description': description,
        'arch': arch,
        'hidden_dim': hidden_dim,
        'activation': activation,
        'epochs': epochs,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'lr': lr,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'epsilon': epsilon,
        'params': params,
        'overfit': overfit,
        'peak_val': peak_val,
        'peak_epoch': peak_epoch,
        'model_size': model_size,
        'best_val': best_val,
        'val_progression': val_progression,
    }


def get_marker_style(row):
    """Get marker style based on experiment type."""
    status = row['status']
    hidden_dim = row['hidden_dim']
    
    if status == 'keep':
        if hidden_dim == 7168:
            return 'o', 'green'
        elif hidden_dim == 8192:
            return 's', 'blue'
        else:
            return 'o', 'green'
    else:
        if hidden_dim == 7168:
            return 'x', 'red'
        elif hidden_dim == 8192:
            return 'd', 'red'
        else:
            return 'x', 'red'


def plot_episode_annotations(ax, df, annotate_all=False):
    """Add annotations for episode points."""
    for _, row in df.iterrows():
        # Only annotate significant points or if annotate_all is True
        if annotate_all or row['val_acc'] >= 0.98 or row['val_acc'] < 0.75:
            label = f'{row["episode"]}'
            if row['hidden_dim']:
                label += f'\n{row["hidden_dim"]}'
            ax.annotate(label, 
                       (row['episode_num'], row['val_acc']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=6, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))


def main():
    # Read file
    with open('results.tsv', 'r') as f:
        lines = f.readlines()
    
    # Parse data
    data = []
    for line in lines[1:]:  # Skip header
        parsed = parse_episode_line(line)
        if parsed:
            data.append(parsed)
    
    df = pd.DataFrame(data)
    
    # Convert episode to numeric
    df['episode_num'] = df['episode'].str.replace('#', '').astype(int)
    
    # Separate kept and discarded
    kept = df[df['status'] == 'keep'].copy()
    discarded = df[df['status'] == 'discard'].copy()
    
    # Find best results
    best_idx = kept['val_acc'].idxmax()
    best_row = kept.loc[best_idx]
    best_val_acc = kept['val_acc'].max()
    
    # Find top 5 kept experiments
    top5 = kept.nlargest(5, 'val_acc')
    
    # Create figure with better layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ===== PANEL 1: Progress Over Time =====
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot all points
    for _, row in df.iterrows():
        marker, color = get_marker_style(row)
        alpha = 0.9 if row['status'] == 'keep' else 0.4
        size = 60 if row['status'] == 'keep' else 40
        ax1.scatter(row['episode_num'], row['val_acc'],
                   c=color, marker=marker, s=size, alpha=alpha, zorder=3 if row['status'] == 'keep' else 2)
    
    # Highlight best
    ax1.scatter(best_row['episode_num'], best_row['val_acc'],
               c='gold', marker='*', s=300, edgecolors='black', linewidths=2, zorder=5)
    ax1.annotate(f'Best: {best_val_acc:.4f}\n{best_row["episode"]}',
                (best_row['episode_num'], best_row['val_acc']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.8))
    
    # Highlight breakthrough episodes
    breakthrough_eps = kept[kept['val_acc'] >= 0.97]
    for _, row in breakthrough_eps.iterrows():
        if row['episode_num'] != best_row['episode_num']:
            ax1.scatter(row['episode_num'], row['val_acc'],
                       c='orange', marker='^', s=100, edgecolors='black', linewidths=1, zorder=4)
    
    # Add trend line
    if len(kept) > 3:
        z = np.polyfit(kept['episode_num'], kept['val_acc'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(kept['episode_num'].min(), kept['episode_num'].max(), 100)
        ax1.plot(x_trend, p(x_trend), '--', color='gray', alpha=0.5, linewidth=1, label='Trend')
    
    ax1.set_xlabel('Episode Number', fontsize=10)
    ax1.set_ylabel('Validation Accuracy', fontsize=10)
    ax1.set_title('MNIST AutoResearch: Validation Accuracy Progress\n(★ = Best, ▲ = Breakthrough > 97%)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.70, 1.00])
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Kept (7168)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Kept (8192)'),
        Line2D([0], [0], marker='x', color='red', markersize=8, label='Discarded'),
        Line2D([0], [0], marker='*', color='gold', markersize=15, label='Best Result'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # ===== PANEL 2: Performance vs Model Size =====
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Scatter plot: params vs val_acc
    for _, row in kept.iterrows():
        if row['params']:
            marker, color = get_marker_style(row)
            ax2.scatter(row['params'] / 1e6, row['val_acc'],
                       c=color, marker=marker, s=80, alpha=0.8, zorder=3)
    
    for _, row in discarded.iterrows():
        if row['params']:
            marker, color = get_marker_style(row)
            ax2.scatter(row['params'] / 1e6, row['val_acc'],
                       c=color, marker=marker, s=50, alpha=0.4, zorder=2)
    
    ax2.scatter(best_row['params'] / 1e6 if best_row['params'] else 31.4, best_row['val_acc'],
               c='gold', marker='*', s=300, edgecolors='black', linewidths=2, zorder=5)
    
    ax2.set_xlabel('Model Size (M params)', fontsize=10)
    ax2.set_ylabel('Validation Accuracy', fontsize=10)
    ax2.set_title('Efficiency: Accuracy vs Model Size', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.70, 1.00])
    
    # ===== PANEL 3: Hyperparameter Analysis =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Group by hidden_dim and activation
    pivot_data = kept.groupby(['hidden_dim', 'activation'])['val_acc'].max().unstack(fill_value=0)
    
    x = np.arange(len(pivot_data.index))
    width = 0.35
    
    if 'GELU' in pivot_data.columns:
        ax3.bar(x - width/2, pivot_data['GELU'], width, label='GELU', color='steelblue', alpha=0.8)
    if 'ReLU' in pivot_data.columns:
        ax3.bar(x + width/2, pivot_data['ReLU'], width, label='ReLU', color='coral', alpha=0.8)
    
    ax3.set_xlabel('Hidden Dimension', fontsize=10)
    ax3.set_ylabel('Best Val Acc', fontsize=10)
    ax3.set_title('Hidden Dim Impact', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(int(h)) if pd.notna(h) else 'N/A' for h in pivot_data.index], fontsize=8)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0.94, 0.992])
    
    # ===== PANEL 4: Batch Size Impact =====
    ax4 = fig.add_subplot(gs[1, 1])
    
    batch_data = kept[kept['batch_size'].notna()].groupby('batch_size')['val_acc'].max()
    
    if len(batch_data) > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(batch_data)))
        bars = ax4.bar(range(len(batch_data)), batch_data.values, color=colors, alpha=0.8)
        ax4.set_xticks(range(len(batch_data)))
        ax4.set_xticklabels([str(int(b)) for b in batch_data.index], fontsize=9)
        ax4.set_xlabel('Batch Size', fontsize=10)
        ax4.set_ylabel('Best Val Acc', fontsize=10)
        ax4.set_title('Batch Size Impact', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([0.94, 0.992])
        
        # Add value labels
        for bar, val in zip(bars, batch_data.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    
    # ===== PANEL 5: Memory Usage =====
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Memory vs performance
    ax5.scatter(kept['memory_gb'], kept['val_acc'],
               c='green', marker='o', s=80, alpha=0.7, label='Kept')
    ax5.scatter(discarded['memory_gb'], discarded['val_acc'],
               c='red', marker='x', s=50, alpha=0.4, label='Discarded')
    
    # Highlight efficient points (high acc, low memory)
    efficient = kept[(kept['val_acc'] >= 0.985) & (kept['memory_gb'] <= 0.7)]
    for _, row in efficient.iterrows():
        ax5.scatter(row['memory_gb'], row['val_acc'],
                   c='gold', marker='*', s=200, edgecolors='black', zorder=5)
    
    ax5.set_xlabel('Memory (GB)', fontsize=10)
    ax5.set_ylabel('Validation Accuracy', fontsize=10)
    ax5.set_title('Memory Efficiency', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0.70, 1.00])
    
    # ===== PANEL 6: Overfitting Analysis =====
    ax6 = fig.add_subplot(gs[2, 0])
    
    kept_with_train = kept[kept['train_acc'].notna()]
    if len(kept_with_train) > 0:
        overfit_gap = kept_with_train['train_acc'] - kept_with_train['val_acc']
        
        colors = ['green' if g < 0.01 else 'orange' if g < 0.05 else 'red' for g in overfit_gap]
        ax6.scatter(kept_with_train['episode_num'], overfit_gap,
                   c=colors, s=60, alpha=0.8)
        
        ax6.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Good (<1%)')
        ax6.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Mild (1-5%)')
        ax6.axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Severe (>5%)')
        
        ax6.set_xlabel('Episode Number', fontsize=10)
        ax6.set_ylabel('Train - Val Accuracy', fontsize=10)
        ax6.set_title('Overfitting Analysis', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8, loc='upper right')
        ax6.grid(True, alpha=0.3)
    
    # ===== PANEL 7: Top Experiments Table =====
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    # Create summary table
    table_data = []
    for _, row in top5.iterrows():
        table_data.append([
            row['episode'],
            f"{row['val_acc']:.4f}",
            f"{row['val_loss']:.4f}" if pd.notna(row['val_loss']) else 'N/A',
            f"{row['memory_gb']:.1f}",
            str(row['hidden_dim']) if pd.notna(row['hidden_dim']) else 'N/A',
            row['activation'],
            str(row['epochs']) if pd.notna(row['epochs']) else 'N/A',
            str(row['batch_size']) if pd.notna(row['batch_size']) else 'N/A'
        ])
    
    table = ax7.table(cellText=table_data,
                     colLabels=['Episode', 'Val Acc', 'Val Loss', 'Mem (GB)', 'Hidden', 'Act', 'Epochs', 'Batch'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Color the header
    for i in range(8):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows based on rank
    for i in range(1, len(table_data) + 1):
        if i == 1:
            color = '#FFD700'  # Gold
        elif i == 2:
            color = '#C0C0C0'  # Silver
        elif i == 3:
            color = '#CD7F32'  # Bronze
        else:
            color = '#E8F4F8'
        for j in range(8):
            table[(i, j)].set_facecolor(color)
    
    ax7.set_title('Top 5 Experiments Summary', fontsize=12, fontweight='bold', pad=20)
    
    # ===== Overall Title =====
    fig.suptitle(f'MNIST AutoResearch Dashboard\nBest Result: {best_val_acc:.4f} ({best_row["episode"]}) | {len(kept)} Kept, {len(discarded)} Discarded',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig('results_plot.png', dpi=150, bbox_inches='tight', facecolor='white')
    print('Plot saved to results_plot.png')
    
    # ===== Print summary =====
    print('\n' + '=' * 70)
    print('SUMMARY STATISTICS')
    print('=' * 70)
    
    print(f"\nBEST RESULT: {best_row['episode']}")
    print(f"  Validation Accuracy: {best_row['val_acc']:.4f} ({best_row['val_acc']*100:.2f}%)")
    print(f"  Validation Loss: {best_row['val_loss']:.4f}" if pd.notna(best_row['val_loss']) else "  Validation Loss: N/A")
    print(f"  Model Size: {best_row['model_size']:.1f}M params" if pd.notna(best_row['model_size']) else "  Model Size: N/A")
    print(f"  Memory: {best_row['memory_gb']:.1f} GB")
    
    # Overfitting check
    best_train = best_row['train_acc'] if pd.notna(best_row['train_acc']) else None
    if best_train:
        overfit = best_train - best_row['val_acc']
        print(f"  Overfitting Gap: {overfit:.4f} (train: {best_train:.4f}, val: {best_row['val_acc']:.4f})")
        if overfit < 0.005:
            print("  -> Excellent: Minimal overfitting!")
        elif overfit < 0.01:
            print("  -> Good: Low overfitting")
        else:
            print("  -> Warning: Overfitting detected")
    
    print(f"\nArchitecture: {best_row['arch']}")
    print(f"  Hidden dimension: {best_row['hidden_dim']}")
    print(f"  Activation: {best_row['activation']}")
    print(f"  Epochs: {best_row['epochs']}")
    print(f"  Batch size: {best_row['batch_size']}")
    print(f"  Learning rate: {best_row['lr']}")
    print(f"  Weight decay: {best_row['weight_decay']}")
    print(f"  Epsilon: {best_row['epsilon']}")
    
    print(f"\nTotal Experiments: {len(df)}")
    print(f"  Kept: {len(kept)} ({len(kept)/len(df)*100:.1f}%)")
    print(f"  Discarded: {len(discarded)} ({len(discarded)/len(df)*100:.1f}%)")
    
    # Best vs discarded gap
    if len(discarded) > 0:
        best_discarded = discarded['val_acc'].max()
        gap = best_val_acc - best_discarded
        print(f"\nBest Discarded: {best_discarded:.4f}")
        print(f"Gap to Best Kept: +{gap:.4f} ({gap*100:.2f}%)")
    
    print('\n' + '=' * 70)
    print('TOP 5 EXPERIMENTS:')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. {row['episode']}: {row['val_acc']:.4f} ({row['activation']}, hidden={row['hidden_dim']}, batch={row['batch_size']})")
    
    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()

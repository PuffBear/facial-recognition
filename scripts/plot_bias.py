import json
import sys
import matplotlib.pyplot as plt

def main(path: str):
    with open(path, 'r') as f:
        stats = json.load(f)
    groups = list(stats.keys())
    accs = [stats[g]['acc'] for g in groups]
    counts = [stats[g]['count'] for g in groups]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    bars = ax1.bar(groups, accs, color='#4C78A8')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Bias: Accuracy by Group')
    for i, b in enumerate(bars):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                 f"{accs[i]:.2f} (n={counts[i]})",
                 ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    out = path.replace('.json', '.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python scripts/plot_bias.py plots/bias_groups.json')
        sys.exit(1)
    main(sys.argv[1])



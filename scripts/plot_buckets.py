import json, sys, os
import matplotlib.pyplot as plt

def load(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_file(path, title):
    obj = load(path)
    models = sorted(obj.keys())
    severities = sorted(next(iter(obj.values())).keys())
    for m in models:
        xs = []
        ys = []
        for s in severities:
            xs.append(s)
            ys.append(obj[m][s]['acc'])
        plt.plot(xs, ys, marker='o', label=m)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out = path.replace('.json', '.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/plot_buckets.py plots/buckets_*.json')
        sys.exit(1)
    for p in sys.argv[1:]:
        title = os.path.basename(p).replace('buckets_', '').replace('.json', '')
        plot_file(p, title)



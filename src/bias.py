from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def compute_group_metrics(preds: List[Tuple[str, str]], groups: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    preds: list of (true_label, pred_label)
    groups: mapping from true_label -> group_name
    Returns: {group: {acc, count}}
    """
    stats = {}
    for t, p in preds:
        g = groups.get(t, 'unknown')
        d = stats.setdefault(g, {'correct': 0, 'count': 0})
        d['count'] += 1
        if t == p:
            d['correct'] += 1
    for g, d in stats.items():
        d['acc'] = (d['correct'] / d['count']) if d['count'] else 0.0
    return stats

def save_group_metrics(stats: Dict[str, Dict[str, float]], outpath: str) -> None:
    with open(outpath, 'w') as f:
        json.dump(stats, f, indent=2)

def plot_group_metrics(stats: Dict[str, Dict[str, float]], out_png: str) -> None:
    groups = list(stats.keys())
    accs = [stats[g]['acc'] for g in groups]
    counts = [stats[g]['count'] for g in groups]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(groups, accs, color='#4C78A8')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Group')
    for i, b in enumerate(bars):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{accs[i]:.2f} (n={counts[i]})", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)

def infer_gender_groups_for_ids(id_to_paths: Dict[str, List[str]]) -> Dict[str, str]:
    """Infer a gender label per identity using one or more sample images via DeepFace.
    Returns mapping: identity -> {'male'|'female'|'unknown'}.
    Requires deepface installed. Falls back to 'unknown' on errors.
    """
    try:
        from deepface import DeepFace  # type: ignore
    except Exception:
        raise ImportError("deepface not installed")

    id_to_group: Dict[str, str] = {}
    for pid, paths in id_to_paths.items():
        pred_counts = defaultdict(int)
        for p in paths[:2]:  # use up to 2 samples per ID
            try:
                res = DeepFace.analyze(img_path=p, actions=['gender'], enforce_detection=False)
                # deepface returns list or dict depending on version
                if isinstance(res, list):
                    res = res[0]
                g = str(res.get('dominant_gender', 'unknown')).lower()
                if g in ('man','male'):
                    pred_counts['male'] += 1
                elif g in ('woman','female'):
                    pred_counts['female'] += 1
                else:
                    pred_counts['unknown'] += 1
            except Exception:
                pred_counts['unknown'] += 1
        group = max(pred_counts, key=lambda k: pred_counts[k]) if pred_counts else 'unknown'
        id_to_group[pid] = group
    return id_to_group

def infer_gender_groups_by_name(labels: List[str]) -> Dict[str, str]:
    """Infer gender from the identity label string using gender-guesser (name-based)."""
    try:
        from gender_guesser.detector import Detector  # type: ignore
    except Exception:
        raise ImportError("gender-guesser not installed")
    det = Detector(case_sensitive=False)
    mapping: Dict[str, str] = {}
    for lab in labels:
        # Extract a likely first name token from label (split on _ and CamelCase boundaries)
        token = lab
        if '_' in token:
            token = token.split('_')[0]
        # If CamelCase, split prefix
        import re
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', token)
        if parts:
            token = parts[0]
        g = det.get_gender(token)
        if g in ("male","mostly_male"):
            mapping[lab] = "male"
        elif g in ("female","mostly_female"):
            mapping[lab] = "female"
        else:
            mapping[lab] = "unknown"
    return mapping



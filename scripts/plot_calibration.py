"""Plot binned confidence versus observed accuracy from saved results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['svg.hashsalt'] = 'calibration'


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / 'results/calibration/summary.json'
ASSETS = ROOT / 'docs/assets'
METHODS = {
    'qwen_hybrid': ('Qwen3.5-9B: mean token probability for chosen action', '#1976a3', 'calibration_qwen.svg'),
    'jev': ('Jev 1.13.0: API probabilities', '#9a4db1', 'calibration_jev.svg'),
}


def save(fig, name: str) -> None:
    output = ASSETS / name
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', metadata={'Date': None})
    fig.savefig(output.with_suffix('.png'), dpi=180, bbox_inches='tight')
    output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines()) + '\n')
    print(output, output.with_suffix('.png'))


def plot_reliability(suites: dict, method: str, title: str, color: str, filename: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), sharex=True, sharey=True)
    for ax, (suite, results) in zip(axes.flat, suites.items()):
        bins = [b for b in results[method]['bins'] if b['count']]
        ax.bar([b['lower'] + .05 for b in bins], [b['accuracy'] for b in bins],
               width=.083, color=color, alpha=.85)
        ax.plot([0, 1], [0, 1], '--', color='#555555', linewidth=1.2, zorder=4)
        ax.set_title(f"{suite} (n={results['jev']['n']:,})", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.18)
    for ax in axes[-1]:
        ax.set_xlabel('Stated confidence (10 bins)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Observed accuracy')
    fig.suptitle(title + ': observed accuracy by confidence', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .955))
    save(fig, filename)
    plt.close(fig)


def main() -> None:
    suites = json.loads(SUMMARY.read_text())['suites']
    for method, (title, color, filename) in METHODS.items():
        plot_reliability(suites, method, title, color, filename)


if __name__ == '__main__':
    main()

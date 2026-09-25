"""Render the README WebShop comparison from the verified result summary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]

def main():
    rows = json.loads((ROOT / 'results/webshop/summary.json').read_text())
    keys = ['qwen', 'jev', 'open_jev']
    assert all(rows[k]['complete'] and rows[k]['completed'] == 500 for k in keys)
    values = [100 * rows[k]['success_rate'] for k in keys]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'webshop-results'})
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor='white')
    fig.subplots_adjust(left=.10, right=.97, bottom=.25, top=.73)
    fig.text(.065, .93, 'Qwen3.5-9B: no additional fine-tuning',
             fontsize=23, fontweight='bold', color='#10233f')
    fig.text(.065, .872, 'Released checkpoint + next-token probabilities',
             fontsize=14, color='#1764bf')
    fig.text(.065, .814, 'WebShop  ·  All 500 test tasks  ·  Full product catalog',
             fontsize=12, color='#526173')
    colors = ['#2375d8', '#8697b2', '#b5c1d2']
    bars = ax.bar(range(3), values, width=.53, color=colors, zorder=3)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel('Task success rate', labelpad=12, color='#37465a')
    ax.set_xticks(range(3), ['Qwen3.5-9B', 'Jev 1.13.0', 'Open-Jev-9B'])
    ax.tick_params(axis='both', length=0, pad=10, labelcolor='#37465a')
    ax.get_xticklabels()[0].set_color('#1764bf')
    ax.get_xticklabels()[0].set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='y', color='#e7ecf2', linewidth=1, zorder=0)
    for bar, key, value in zip(bars, keys, values):
        x = bar.get_x() + bar.get_width()/2
        ax.text(x, value+.7, f'{value:.1f}%', ha='center', va='bottom',
                fontsize=21, fontweight='bold', color='#10233f')
        ax.text(x, value-2.4, f"{rows[key]['successes']} / 500", ha='center',
                color='white' if key=='qwen' else '#172c49', fontsize=11)
    ax.text(0, -.20, 'NO ADDITIONAL FT', transform=ax.get_xaxis_transform(),
            ha='center', fontsize=11, fontweight='bold', color='#1764bf',
            bbox={'boxstyle':'round,pad=.4', 'facecolor':'#eaf3ff', 'edgecolor':'none'})
    fig.text(.065, .065, 'Qwen and Jev differ by one successful task; this does not establish a success-rate advantage.',
             fontsize=10, color='#526173')
    target = ROOT / 'docs/assets/webshop_success'
    fig.savefig(target.with_suffix('.svg'), metadata={'Date': None}, facecolor='white')
    fig.savefig(target.with_suffix('.png'), dpi=180, facecolor='white')
    plt.close(fig)

if __name__ == '__main__':
    main()

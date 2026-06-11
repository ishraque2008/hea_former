"""
Publication-quality figures for HEA Transformer paper.

F1  architecture schematic (text-based, matplotlib)
F2  tokenization illustration
F3  SRO parity scatter
F4  attention map heatmap
F5  ablation bar chart
F6  per-scenario MAE
F7  training curves
F8  Warren-Cowley SRO matrix
F9  confusion matrix
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

CANTOR = ['Cr', 'Mn', 'Fe', 'Co', 'Ni']
EL_COL = {'Cr':'#E64B35','Mn':'#4DBBD5','Fe':'#00A087',
           'Co':'#3C5488','Ni':'#F39B7F'}
PH_COL = ['#2196F3','#FF9800','#F44336']
PH_NM  = ['Disordered','Weak order','Strong order']

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name, dpi=150):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig] {p}")
    return p


# ---------------------------------------------------------------------------
# F1: Architecture schematic
# ---------------------------------------------------------------------------
def plot_architecture() -> str:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off')

    def box(x, y, w, h, color, text, fs=9):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.1',
            fc=color, ec='#333', linewidth=1.2))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', wrap=True,
                multialignment='center')

    def arrow(x1, x2, y, label='', color='#555'):
        ax.annotate('', xy=(x2,y), xytext=(x1,y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if label:
            ax.text((x1+x2)/2, y+0.15, label, ha='center',
                    fontsize=7, color=color)

    # Input blocks
    box(0.1, 2.5, 1.6, 1.5, '#E3F2FD', 'FCC\nSupercell\n(N sites)', 8)
    box(0.1, 0.5, 1.6, 1.5, '#FFF3E0', 'HRTEM /\nSTEM\npatches', 8)
    arrow(1.7, 2.5, 3.25)
    arrow(1.7, 2.5, 1.25)

    # Tokenizer
    box(2.5, 2.5, 1.8, 1.5, '#E8F5E9',
        'Site-Occupancy\nTokenizer\n(element+env)', 8)
    arrow(4.3, 5.0, 3.25)

    # CNN
    box(2.5, 0.5, 1.8, 1.5, '#FCE4EC',
        'CNN / ViT\nImage\nEncoder', 8)
    arrow(4.3, 5.0, 1.25)

    # Transformer encoder
    box(5.0, 1.5, 2.2, 2.5, '#EDE7F6',
        'Transformer\nEncoder\n(L layers\nMHA + FF)', 9)
    arrow(7.2, 8.2, 3.25, 'fuse')
    arrow(7.2, 8.2, 1.25)

    # Fusion
    box(8.2, 1.8, 1.5, 1.8, '#FFFDE7', 'Fusion\nlayer', 8)
    arrow(9.7, 10.2, 2.7, '')

    # Heads
    box(10.2, 3.2, 1.6, 1.0, '#E0F7FA', 'SRO\nhead', 8)
    box(10.2, 1.9, 1.6, 1.0, '#F3E5F5', 'Phase\nhead', 8)
    box(10.2, 0.6, 1.6, 1.0, '#FBE9E7', 'MLM\nhead', 8)
    ax.annotate('', xy=(10.2,3.7), xytext=(9.7,2.7),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
    ax.annotate('', xy=(10.2,2.4), xytext=(9.7,2.7),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
    ax.annotate('', xy=(10.2,1.1), xytext=(9.7,2.7),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

    ax.set_title('HEAFormer Architecture', fontsize=13, fontweight='bold', pad=8)
    fig.tight_layout()
    return _save(fig, 'F1_architecture.png')


# ---------------------------------------------------------------------------
# F2: Tokenisation illustration
# ---------------------------------------------------------------------------
def plot_tokenization(occ, token_ids, mask, n_show=24) -> str:
    n = min(n_show, len(occ))
    fig, axes = plt.subplots(2, 1, figsize=(min(n*0.5+1, 14), 3.5))

    for row, (ids, is_input) in enumerate([(occ, False), (token_ids, True)]):
        ax = axes[row]
        for i in range(n):
            t  = int(ids[i])
            m  = mask[i] if is_input else False
            if m:
                color = '#B0BEC5'; label = '[M]'; ec = 'red'; lw = 2.0
            elif 0 <= t < len(CANTOR):
                color = EL_COL[CANTOR[t]]; label = CANTOR[t]; ec = 'k'; lw = 0.5
            else:
                color = '#CFD8DC'; label = '?'; ec = 'k'; lw = 0.5
            ax.bar(i, 1, color=color, edgecolor=ec, linewidth=lw, width=0.9)
            ax.text(i, 0.5, label, ha='center', va='center',
                    fontsize=6.5, fontweight='bold',
                    color='white' if not m else 'red')
        ax.set_xlim(-0.5, n-0.5); ax.set_ylim(0,1)
        ax.set_yticks([])
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(i) for i in range(n)], fontsize=5.5)
        ax.set_title('True site occupancy' if not is_input
                     else f'Masked input  ({mask[:n].sum()}/{n} masked, [M]=red)',
                     fontsize=9, loc='left')

    handles = [mpatches.Patch(color=EL_COL[e], label=e) for e in CANTOR]
    axes[0].legend(handles=handles, loc='upper right',
                   ncol=5, fontsize=7, framealpha=0.85)
    fig.tight_layout(pad=0.6)
    return _save(fig, 'F2_tokenization.png')


# ---------------------------------------------------------------------------
# F3: SRO parity plot
# ---------------------------------------------------------------------------
def plot_sro_parity(sro_pred, sro_true, model_name='HEA Transformer',
                     mae=None, r2=None) -> str:
    fp = sro_pred.ravel(); ft = sro_true.ravel()
    if len(fp) > 8000:
        idx = np.random.choice(len(fp), 8000, replace=False)
        fp, ft = fp[idx], ft[idx]
    lim = max(abs(ft).max(), abs(fp).max()) * 1.1
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ft, fp, alpha=0.25, s=5, c='#3C5488', rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.5, label='Ideal')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel(r'True $\alpha_{ij}^{(m)}$', fontsize=11)
    ax.set_ylabel(r'Predicted $\alpha_{ij}^{(m)}$', fontsize=11)
    ttl = model_name
    if mae is not None: ttl += f'\nMAE={mae:.4f}'
    if r2  is not None: ttl += f'   R²={r2:.4f}'
    ax.set_title(ttl, fontsize=10); ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    return _save(fig, f'F3_parity_{model_name[:12].replace(" ","_")}.png')


# ---------------------------------------------------------------------------
# F4: Attention heatmap
# ---------------------------------------------------------------------------
def plot_attention(attn, occ, layer=0, head=0, n_show=20, title='') -> str:
    A = attn[head, :n_show, :n_show]
    labels = [CANTOR[int(o)] for o in occ[:n_show]]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(A, cmap='Blues', vmin=0, aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention weight')
    ax.set_xticks(range(n_show)); ax.set_yticks(range(n_show))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    for tick, lbl in zip(ax.get_xticklabels(), labels):
        tick.set_color(EL_COL.get(lbl, 'k'))
    for tick, lbl in zip(ax.get_yticklabels(), labels):
        tick.set_color(EL_COL.get(lbl, 'k'))
    ax.set_xlabel('Key site', fontsize=10); ax.set_ylabel('Query site', fontsize=10)
    ax.set_title(f'Attention weights  L{layer+1} H{head+1}\n{title}', fontsize=10)
    fig.tight_layout()
    return _save(fig, f'F4_attn_L{layer+1}_H{head+1}.png')


# ---------------------------------------------------------------------------
# F5: Ablation bar chart
# ---------------------------------------------------------------------------
def plot_ablation(results: dict, metric='mae', task='SRO regression') -> str:
    names, vals = [], []
    for n, m in results.items():
        if isinstance(m, dict) and metric in m:
            names.append(n); vals.append(m[metric])
    if not names:
        return ''
    asc = 'mae' in metric or 'rmse' in metric
    order = np.argsort(vals if asc else [-v for v in vals])
    names = [names[i] for i in order]
    vals  = [vals[i]  for i in order]
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(names)))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45*len(names))))
    bars = ax.barh(range(len(names)), vals, color=colors,
                    edgecolor='k', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(metric.upper(), fontsize=11)
    ax.set_title(f'{task} — model comparison', fontsize=12)
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals)*0.01, bar.get_y()+bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=8)
    ax.grid(axis='x', ls='--', alpha=0.3)
    fig.tight_layout()
    return _save(fig, f'F5_ablation_{metric}_{task[:6]}.png')


# ---------------------------------------------------------------------------
# F6: Per-scenario MAE
# ---------------------------------------------------------------------------
def plot_scenario_mae(scenario_maes: dict, model_names: list) -> str:
    scenarios = list(list(scenario_maes.values())[0].keys())
    x = np.arange(len(scenarios))
    w = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, mn in enumerate(model_names):
        v = [scenario_maes[mn].get(s, np.nan) for s in scenarios]
        offset = (i - len(model_names)/2 + 0.5) * w
        ax.bar(x + offset, v, w*0.9, label=mn,
               color=plt.cm.Set2(i/max(len(model_names)-1,1)),
               edgecolor='k', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_','\n') for s in scenarios], fontsize=8)
    ax.set_ylabel('SRO MAE', fontsize=11)
    ax.set_title('Per-scenario SRO MAE', fontsize=12)
    ax.legend(fontsize=8); ax.grid(axis='y', ls='--', alpha=0.3)
    fig.tight_layout()
    return _save(fig, 'F6_scenario_mae.png')


# ---------------------------------------------------------------------------
# F7: Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict) -> str:
    keys = [('pretrain_loss', None, 'MLM pretrain loss'),
            ('train_loss',  'val_loss',   'Fine-tune loss'),
            ('train_mae',   'val_mae',    'SRO MAE')]
    n_plots = sum(1 for tk, _, _ in keys
                  if tk in history and history[tk])
    if n_plots == 0:
        return ''
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5*n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    j = 0
    for tk, vk, label in keys:
        if tk not in history or not history[tk]:
            continue
        ax = axes[j]; j += 1
        ep = range(1, len(history[tk])+1)
        ax.plot(ep, history[tk], 'b-o', ms=3, lw=1.5, label='Train')
        if vk and vk in history and history[vk]:
            ax.plot(range(1, len(history[vk])+1), history[vk],
                    'r-s', ms=3, lw=1.5, label='Val')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.3)
    fig.suptitle('HEAFormer Training History', fontsize=12,
                  fontweight='bold', y=1.01)
    fig.tight_layout()
    return _save(fig, 'F7_training_curves.png')


# ---------------------------------------------------------------------------
# F8: SRO matrix heatmap
# ---------------------------------------------------------------------------
def plot_sro_matrix(alpha, shell=0, title='') -> str:
    A = alpha[shell]; ns = A.shape[0]
    vmax = max(abs(A).max(), 0.25)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    plt.colorbar(im, ax=ax, label=r'$\alpha_{ij}$')
    ax.set_xticks(range(ns)); ax.set_yticks(range(ns))
    ax.set_xticklabels(CANTOR, fontsize=10)
    ax.set_yticklabels(CANTOR, fontsize=10)
    ax.set_xlabel('Neighbor species $j$', fontsize=11)
    ax.set_ylabel('Centre species $i$',   fontsize=11)
    for i in range(ns):
        for j in range(ns):
            ax.text(j, i, f'{A[i,j]:+.2f}', ha='center', va='center',
                    fontsize=7.5,
                    color='white' if abs(A[i,j]) > 0.5*vmax else 'black')
    ax.set_title(fr'Warren-Cowley SRO (shell {shell+1})'
                 + (f'\n{title}' if title else ''), fontsize=11)
    fig.tight_layout()
    tag = title[:10].replace(' ', '_') if title else 'sample'
    return _save(fig, f'F8_sro_matrix_sh{shell+1}_{tag}.png')


# ---------------------------------------------------------------------------
# F9: Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion(cm_list, title='') -> str:
    cm = np.array(cm_list) if isinstance(cm_list, list) else cm_list
    if cm.sum() == 0:
        return ''
    cm_n = cm.astype(float) / cm.sum(1, keepdims=True).clip(1e-9)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Normalised')
    labels = ['Disord.', 'Weak\norder', 'Strong\norder']
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            n = cm[i,j] if i < cm.shape[0] and j < cm.shape[1] else 0
            v = cm_n[i,j] if i < cm_n.shape[0] and j < cm_n.shape[1] else 0
            ax.text(j, i, f'{v:.2f}\n({n})', ha='center', va='center',
                    fontsize=8, color='white' if v > 0.55 else 'black')
    ax.set_title(f'Phase confusion matrix\n{title}', fontsize=11)
    fig.tight_layout()
    return _save(fig, 'F9_confusion.png')

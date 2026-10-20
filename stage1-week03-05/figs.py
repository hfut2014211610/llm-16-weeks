"""生成讲义插图。python figs.py （图 3 需要先跑完 ablation.py）"""
import json, math, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np

for f in ['NotoSansSC-Regular.otf', 'NotoSansSC-Bold.otf', 'NotoSansSC-Regular.ttf', 'NotoSansSC-Bold.ttf']:
    pth = os.path.expanduser('~/.fonts/' + f)
    if os.path.exists(pth): font_manager.fontManager.addfont(pth)
plt.rcParams['font.family'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 图 1：GPT-2 块 vs Llama 块 ----------
def box(ax, y, text, color, h=0.62, w=3.2, x=0.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', fc=color, ec='#555', lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=10.5)
def arrow(ax, y0, y1, x=2.0):
    ax.annotate('', xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle='->', color='#444', lw=1.2))

def column(ax, title, items):
    ax.set_xlim(0, 4); ax.set_ylim(0, 10.6); ax.axis('off'); ax.set_title(title, fontsize=13, fontweight='bold')
    y = 0.2
    for i, (t, c) in enumerate(items):
        box(ax, y, t, c)
        if i < len(items) - 1: arrow(ax, y + 0.66, y + 0.98)
        y += 1.02

same, diff = '#e8eef7', '#ffe3c2'
fig, axes = plt.subplots(1, 2, figsize=(9, 6.6))
column(axes[0], 'GPT-2 / 第 1 批 gpt.py', [
    ('字向量 + 学习型位置向量', diff), ('LayerNorm（减均值，有 bias）', diff),
    ('多头注意力 MHA（K/V 头数 = Q 头数）', diff), ('+ 残差', same), ('LayerNorm', diff),
    ('FFN：Linear→ReLU→Linear（4d）', diff), ('+ 残差', same), ('× N 层 → LayerNorm', diff),
    ('lm_head（GPT-2 共享字向量表，gpt.py 未共享）', diff), ('logits', same)])
column(axes[1], 'Llama / llama.py', [
    ('只有字向量（位置交给 RoPE）', diff), ('RMSNorm（不减均值，无 bias）', diff),
    ('GQA 注意力 + RoPE 旋转 Q/K', diff), ('+ 残差', same), ('RMSNorm', diff),
    ('FFN：SwiGLU（≈8/3·d，三张表）', diff), ('+ 残差', same), ('× N 层 → RMSNorm', diff),
    ('lm_head（Llama-2/3 8B 不共享）', diff), ('logits', same)])
fig.text(0.5, 0.01, '橙色 = 两者不同；蓝色 = 相同。两者都是 Pre-norm（先归一化再进子层），Llama 所有 Linear 都没有 bias。',
         ha='center', fontsize=9.5, color='#444')
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig('fig1_blocks.png', dpi=170); plt.close()

# ---------- 图 2：RoPE ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
ax = axes[0]; v = np.array([1.0, 0.4]); step = math.radians(25)
cols = plt.cm.viridis(np.linspace(0, 0.9, 6))
for m in range(6):
    c, s = math.cos(m * step), math.sin(m * step)
    w = np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])
    ax.annotate('', xy=w, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=cols[m], lw=2))
    ax.text(w[0] * 1.12, w[1] * 1.12, f'位置{m}', fontsize=9, color=cols[m], ha='center')
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.3, 1.3); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_title('同一对数 (x0,x1)，位置每 +1 就多转一个固定角度\n长度不变，只改方向', fontsize=10.5)

ax = axes[1]
def dot_vs_dist(hd, theta, D):
    inv = 1.0 / (theta ** (np.arange(0, hd, 2) / hd))
    # q = k = 全 1 向量时，q_m·k_n = 2·Σ cos((m-n)·inv)
    return np.array([2 * np.cos(d * inv).sum() for d in D])
D = np.arange(0, 1025)
for theta, c in [(100, '#d62728'), (10000, '#1f77b4'), (500000, '#2ca02c')]:
    ax.plot(D, dot_vs_dist(128, theta, D) / 128, label=f'rope_theta={theta:g}', color=c, lw=1.2)
ax.set_xlabel('相对距离 m−n（位置）'); ax.set_ylabel('点积 / head_dim')
ax.set_title('head_dim=128，同一个 q、k 放在不同距离\n距离越远点积越小；theta 越大衰减越慢', fontsize=10.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig2_rope.png', dpi=170); plt.close()

# ---------- 图 4：KV cache ----------
L, hd, ctx = 32, 128, 8192
labels, vals = ['MHA\n32 个 KV 头', 'GQA（实际）\n8 个 KV 头', 'MQA\n1 个 KV 头'], []
for kv in (32, 8, 1): vals.append(2 * L * kv * hd * 2 * ctx / 2 ** 30)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
bars = ax.bar(labels, vals, color=['#bbb', '#1f77b4', '#9ecae1'])
for b, v_ in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v_ + 0.06, f'{v_:.3g} GiB', ha='center')
ax.set_ylabel('GiB'); ax.set_ylim(0, 4.6)
ax.set_title('Llama-3-8B，8192 上下文，fp16，单条请求的 KV cache', fontsize=10.5)
plt.tight_layout(); plt.savefig('fig4_kvcache.png', dpi=170); plt.close()

# ---------- 图 3：消融 loss 曲线 ----------
runs = [('A_gpt', 'A GPT 基线'), ('B_rms', 'B +RMSNorm'), ('C_rope', 'C +RoPE'),
        ('D_swiglu', 'D +SwiGLU'), ('E_gqa', 'E +GQA'), ('F_llama', 'F 完整 Llama')]
if all(os.path.exists(f'results/{r}.json') for r, _ in runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for r, lab in runs:
        cv = np.array(json.load(open(f'results/{r}.json'))['curve'])
        ls = '-' if r in ('A_gpt', 'F_llama') else '--'
        lw = 2 if r in ('A_gpt', 'F_llama') else 1.1
        axes[0].plot(cv[:, 0], cv[:, 2], ls, lw=lw, label=lab)
        axes[1].plot(cv[4:, 0], cv[4:, 2], ls, lw=lw, label=lab)
    axes[0].set_title('val loss，全程', fontsize=10.5); axes[1].set_title('val loss，第 1000 步以后放大', fontsize=10.5)
    for a in axes: a.set_xlabel('训练步数'); a.grid(alpha=0.3)
    axes[0].set_ylabel('val loss'); axes[1].legend(fontsize=8.5)
    plt.tight_layout(); plt.savefig('fig3_ablation.png', dpi=170); plt.close()
    print('fig3 ok')
print('done')

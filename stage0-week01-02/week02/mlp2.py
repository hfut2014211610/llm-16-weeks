# mlp2.py —— 第 2 周补强：字符级 MLP 造名字（makemore 第 2–4 课精神）
# 运行：python mlp2.py        （CPU，几分钟；首次会自动下载 names.txt）
# 输出：终端打印各阶段 loss + 生成的名字；当前目录生成 fig1~fig5 五张 png
import os, math, random, time, urllib.request
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

torch.set_num_threads(max(1, os.cpu_count() or 1))
T0 = time.time()

# ---------------- 中文字体（找不到就用默认字体，不影响训练） ----------------
here = os.path.dirname(os.path.abspath(__file__))
for d in [os.path.expanduser('~/.fonts'), os.path.join(here, '..', '..', 'work', 'fonts')]:
    p = os.path.join(d, 'NotoSansSC-Regular.ttf')
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        b = os.path.join(d, 'NotoSansSC-Bold.ttf')
        if os.path.exists(b): font_manager.fontManager.addfont(b)
        plt.rcParams['font.family'] = 'Noto Sans SC'
        break
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 数据 ----------------
if not os.path.exists('names.txt'):
    print('下载 names.txt ...')
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt', 'names.txt')
words = open('names.txt').read().splitlines()
chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}; stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
V = len(stoi)                 # 27 = 26 个字母 + '.'
block_size = 3                # 看前 3 个字符

def build_dataset(ws):
    X, Y = [], []
    for w in ws:
        ctx = [0] * block_size
        for ch in w + '.':
            X.append(ctx); Y.append(stoi[ch])
            ctx = ctx[1:] + [stoi[ch]]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42); random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
print(f'名字 {len(words)} 个，字表 {V}；样本数 train {len(Xtr)} / dev {len(Xdev)} / test {len(Xte)}')

def section(t): print('\n' + '=' * 8, t, '=' * 8)

# ================= 第 1 章：bigram 基线 =================
section('1. bigram 基线（只看前 1 个字符）')
N = torch.zeros((V, V))
N.index_put_((Xtr[:, -1], Ytr), torch.ones(len(Ytr)), accumulate=True)   # 只用最后一个字符
P = (N + 1) / (N + 1).sum(1, keepdim=True)
bigram_tr = -P[Xtr[:, -1], Ytr].log().mean().item()
bigram_dev = -P[Xdev[:, -1], Ydev].log().mean().item()
print(f'bigram  train {bigram_tr:.4f}  dev {bigram_dev:.4f}')
print(f'瞎猜（均匀分布）的 loss = -ln(1/27) = {-math.log(1/27):.4f}')

# ================= 通用 MLP =================
n_embd, n_hidden = 10, 200
ACTS = {'tanh': torch.tanh, 'relu': F.relu, 'gelu': F.gelu, 'silu': F.silu}

def make_mlp(init='kaiming', act='tanh', bn=False, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    fan_in = n_embd * block_size
    gain = {'tanh': 5/3, 'relu': 2**0.5, 'gelu': 2**0.5, 'silu': 2**0.5}[act]
    p = {'C': torch.randn((V, n_embd), generator=g)}
    if init == 'naive':          # 不做任何缩放
        p['W1'] = torch.randn((fan_in, n_hidden), generator=g)
        p['b1'] = torch.randn(n_hidden, generator=g)
        p['W2'] = torch.randn((n_hidden, V), generator=g)
        p['b2'] = torch.randn(V, generator=g)
    else:                        # Kaiming 缩放 + 输出层缩小
        p['W1'] = torch.randn((fan_in, n_hidden), generator=g) * gain / fan_in**0.5
        p['b1'] = torch.randn(n_hidden, generator=g) * 0.01
        p['W2'] = torch.randn((n_hidden, V), generator=g) * 0.01
        p['b2'] = torch.zeros(V)
    if bn:
        p['bngain'] = torch.ones(n_hidden); p['bnbias'] = torch.zeros(n_hidden)
    for t in p.values(): t.requires_grad = True
    m = {'p': p, 'act': act, 'bn': bn,
         'bnmean': torch.zeros(n_hidden), 'bnstd': torch.ones(n_hidden)}
    return m

def forward(m, X, train=False):
    p = m['p']
    emb = p['C'][X].view(X.shape[0], -1)
    hpre = emb @ p['W1'] + p['b1']
    if m['bn']:
        if train:                               # 训练：用这一批自己的均值/标准差
            mu, sd = hpre.mean(0, keepdim=True), hpre.std(0, keepdim=True)
            with torch.no_grad():               # 顺手更新“滑动平均”，留给推理用
                m['bnmean'] = 0.999 * m['bnmean'] + 0.001 * mu.squeeze(0)
                m['bnstd'] = 0.999 * m['bnstd'] + 0.001 * sd.squeeze(0)
        else:                                   # 推理：用训练时攒下的滑动平均
            mu, sd = m['bnmean'], m['bnstd']
        hpre = p['bngain'] * (hpre - mu) / (sd + 1e-5) + p['bnbias']
    h = ACTS[m['act']](hpre)
    return h @ p['W2'] + p['b2'], hpre, h

@torch.no_grad()
def evaluate(m, X, Y):
    return F.cross_entropy(forward(m, X)[0], Y).item()

def train(m, steps, lr_fn, bs=32, seed=42, log_every=None):
    g = torch.Generator().manual_seed(seed)
    params = list(m['p'].values())
    lossi = []
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (bs,), generator=g)
        loss = F.cross_entropy(forward(m, Xtr[ix], train=True)[0], Ytr[ix])
        for q in params: q.grad = None
        loss.backward()
        lr = lr_fn(i)
        with torch.no_grad():
            for q in params: q -= lr * q.grad
        lossi.append(loss.item())
        if log_every and i % log_every == 0: print(f'  step {i:6d}  batch loss {loss.item():.4f}')
    return lossi

def sample(m, n=10, seed=2147483647 + 10):
    g = torch.Generator().manual_seed(seed)
    out = []
    with torch.no_grad():
        for _ in range(n):
            ctx, s = [0] * block_size, []
            while True:
                probs = F.softmax(forward(m, torch.tensor([ctx]))[0], dim=1)
                ix = torch.multinomial(probs, 1, generator=g).item()
                ctx = ctx[1:] + [ix]
                if ix == 0: break
                s.append(itos[ix])
            out.append(''.join(s))
    return out

STEPS = 30000
lr_sched = lambda i: 0.1 if i < 20000 else 0.01

# ================= 第 2 章：学习率扫描 + 小批量 =================
section('2. 学习率扫描（1000 步，lr 从 1e-3 指数增长到 1）')
m = make_mlp('kaiming')
lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre
lossi = train(m, 1000, lambda i: lrs[i].item())
sm = torch.tensor(lossi).view(-1, 20).mean(1)             # 每 20 步平均一下，曲线更平滑
lre_sm = lre.view(-1, 20).mean(1)
best = 10 ** lre_sm[sm.argmin()].item()
print(f'平滑后 loss 最低处的 lr ≈ {best:.3f}（10^{math.log10(best):.2f}）')
for e in [-3, -2, -1.5, -1, -0.5, 0]:
    k = (lre_sm - e).abs().argmin()
    print(f'  lr=10^{e:<5} 附近平均 loss {sm[k]:.3f}')
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(lre, lossi, alpha=0.25, color='gray', label='每一步')
ax.plot(lre_sm, sm, color='C0', lw=2, label='20 步平均')
ax.axvline(-1, color='C3', ls='--', lw=1); ax.text(-0.98, max(lossi) * 0.9, '选 lr=0.1', color='C3')
ax.set_xlabel('学习率的指数  log10(lr)'); ax.set_ylabel('batch loss')
ax.set_title('图1  学习率扫描：先降后升，选“谷底偏左”'); ax.legend(); fig.tight_layout()
fig.savefig('fig1_lr_sweep.png', dpi=150); plt.close(fig)

# 小批量 vs 整批：一步要多久
m = make_mlp('kaiming')
params = list(m['p'].values())
for name, X, Y in [('整批 %d 条' % len(Xtr), Xtr, Ytr), ('小批 32 条', Xtr[:32], Ytr[:32])]:
    t = time.time()
    for _ in range(5):
        loss = F.cross_entropy(forward(m, X)[0], Y)
        for q in params: q.grad = None
        loss.backward()
    print(f'  {name}：一步 {(time.time() - t) / 5 * 1000:.1f} ms')

# ================= 第 3 章：初始化 =================
section('3. 初始化：naive vs Kaiming')
res = {}
for init in ['naive', 'kaiming']:
    m = make_mlp(init)
    with torch.no_grad():
        logits, hpre, h = forward(m, Xtr[:1000])
        l0 = F.cross_entropy(logits, Ytr[:1000]).item()
        sat = (h.abs() > 0.99).float().mean().item()
        dead = ((h.abs() > 0.99).all(0)).sum().item()
    print(f'[{init}] 初始 loss {l0:.4f}；logits 绝对值均值 {logits.abs().mean():.2f}；'
          f'hpre 标准差 {hpre.std():.2f}；|h|>0.99 的比例 {sat:.1%}；对 1000 个样本全饱和的神经元 {dead}/{n_hidden}')
    res[init] = dict(l0=l0, hpre=hpre.flatten(), h=h.flatten(), sat=sat)
    t = time.time()
    lossi = train(m, STEPS, lr_sched, log_every=10000)
    res[init].update(lossi=lossi, tr=evaluate(m, Xtr, Ytr), dev=evaluate(m, Xdev, Ydev), model=m)
    print(f'[{init}] 训练 {STEPS} 步后  train {res[init]["tr"]:.4f}  dev {res[init]["dev"]:.4f}  '
          f'（前 100 步平均 batch loss {sum(lossi[:100]) / 100:.3f}，用时 {time.time() - t:.0f}s）')

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, init, title in zip(axes, ['naive', 'kaiming'], ['修前：naive 初始化', '修后：Kaiming 缩放']):
    ax.hist(res[init]['h'].numpy(), bins=50, color='C3' if init == 'naive' else 'C2')
    ax.set_title(f'{title}\n|h|>0.99 占 {res[init]["sat"]:.1%}')
    ax.set_xlabel('tanh 输出 h')
fig.suptitle('图2  隐藏层激活直方图（训练前，1000 个样本 × 200 个神经元）')
fig.tight_layout(); fig.savefig('fig2_tanh_hist.png', dpi=150); plt.close(fig)

# 最终模型（Kaiming 版）在 test 上只看一次
final = res['kaiming']['model']
test_loss = evaluate(final, Xte, Yte)
print(f'[最终模型] test {test_loss:.4f}')
print('生成的名字：', ', '.join(sample(final, 12)))

# ================= 第 4 章：激活函数 =================
section('4. 激活函数对比（同一 MLP，Kaiming 缩放，各训练 15000 步）')
x = torch.linspace(-4, 4, 401, requires_grad=True)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for name, f in ACTS.items():
    y = f(x); d, = torch.autograd.grad(y.sum(), x)
    axes[0].plot(x.detach(), y.detach(), label=name, lw=2)
    axes[1].plot(x.detach(), d, label=name, lw=2)
for v in [-3, -1, 0, 1, 3]:
    xs = torch.tensor(float(v), requires_grad=True)
    ds = [f'{n}={torch.autograd.grad(f(xs), xs)[0].item():.3f}' for n, f in ACTS.items()]
    print(f'  x={v:>2} 处导数：', '  '.join(ds))
axes[0].set_title('函数本身'); axes[1].set_title('导数（梯度能传回去多少）')
for ax in axes: ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5); ax.legend()
fig.suptitle('图3  四种激活函数及其导数'); fig.tight_layout()
fig.savefig('fig3_activations.png', dpi=150); plt.close(fig)

act_res = {}
for act in ACTS:
    m = make_mlp('kaiming', act=act)
    train(m, 15000, lambda i: 0.1 if i < 10000 else 0.01)
    with torch.no_grad():
        h = forward(m, Xtr[:1000])[2]
        zero = (h <= 0).float().mean().item() if act == 'relu' else float('nan')
    act_res[act] = (evaluate(m, Xtr, Ytr), evaluate(m, Xdev, Ydev))
    extra = f'；训练后 h=0 的比例 {zero:.1%}' if act == 'relu' else ''
    print(f'  {act:5s} train {act_res[act][0]:.4f}  dev {act_res[act][1]:.4f}{extra}')

# ================= 第 5 章：BatchNorm =================
section('5. BatchNorm：故意用 naive 的 W1（不缩放），看 BN 能不能救回来')
bn_res = {}
for bn in [False, True]:
    m = make_mlp('kaiming', bn=bn)
    with torch.no_grad():                      # 只把 W1 换成“没缩放”的版本
        g = torch.Generator().manual_seed(1)
        m['p']['W1'].copy_(torch.randn(m['p']['W1'].shape, generator=g))
        logits, hpre, h = forward(m, Xtr[:1000], train=True)
        sat = (h.abs() > 0.99).float().mean().item()
    tag = '有 BN' if bn else '无 BN'
    print(f'[{tag}] 训练前 hpre 标准差 {hpre.std():.2f}；|h|>0.99 比例 {sat:.1%}')
    train(m, STEPS, lr_sched)
    with torch.no_grad():
        h_after = forward(m, Xtr[:1000], train=False)[2]
        sat_after = (h_after.abs() > 0.99).float().mean().item()
    bn_res[bn] = dict(h0=h.flatten(), h1=h_after.flatten(), sat0=sat, sat1=sat_after,
                      tr=evaluate(m, Xtr, Ytr), dev=evaluate(m, Xdev, Ydev), model=m)
    print(f'[{tag}] 训练 {STEPS} 步后 train {bn_res[bn]["tr"]:.4f}  dev {bn_res[bn]["dev"]:.4f}；'
          f'训练后 |h|>0.99 比例 {sat_after:.1%}')

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, bn in zip(axes, [False, True]):
    ax.hist(bn_res[bn]['h0'].numpy(), bins=50, color='C2' if bn else 'C3')
    ax.set_title(f'{"有 BN" if bn else "无 BN"}（W1 未缩放）\n|h|>0.99 占 {bn_res[bn]["sat0"]:.1%}')
    ax.set_xlabel('tanh 输出 h')
fig.suptitle('图4  同样糟糕的 W1，加 BatchNorm 前后的激活分布（训练前）')
fig.tight_layout(); fig.savefig('fig4_batchnorm.png', dpi=150); plt.close(fig)

# 训练 / 推理两种模式的区别
mbn = bn_res[True]['model']
saved = (mbn['bnmean'].clone(), mbn['bnstd'].clone())   # 演示时别污染滑动平均
with torch.no_grad():
    x1 = Xdev[:1]
    a = forward(mbn, x1, train=False)[0]
    b = forward(mbn, Xdev[:64], train=False)[0][:1]
    c = forward(mbn, Xdev[:64], train=True)[0][:1]
    d = forward(mbn, Xdev[64:128].clone().index_copy_(0, torch.tensor([0]), x1), train=True)[0][:1]
print('同一个样本的 logits 前 3 个：')
print('  推理模式，单独喂       ', [round(v, 3) for v in a[0, :3].tolist()])
print('  推理模式，混在 64 条里 ', [round(v, 3) for v in b[0, :3].tolist()])
print('  训练模式，和第 1 批 64 条一起', [round(v, 3) for v in c[0, :3].tolist()])
print('  训练模式，和另一批 64 条一起', [round(v, 3) for v in d[0, :3].tolist()])
mbn['bnmean'], mbn['bnstd'] = saved
print('BN 模型生成的名字（推理模式）：', ', '.join(sample(mbn, 8)))

# 图5：LayerNorm vs BatchNorm 归一化的方向
h = torch.tensor([[2., 4., 6., 8.], [1., 1., 1., 5.], [10., 0., 10., 0.]])
bn_out = (h - h.mean(0)) / h.std(0, unbiased=False)
ln_out = F.layer_norm(h, (4,))
print('\n演示矩阵 h（3 个样本 × 4 个特征）')
print('  BatchNorm 按列（跨样本）归一化：\n', bn_out.round(decimals=2))
print('  LayerNorm 按行（每个样本自己）归一化：\n', ln_out.round(decimals=2))
fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
for ax, M, t in zip(axes, [h, bn_out, ln_out], ['原始 h（行=样本，列=特征）', 'BatchNorm：每一列均值0', 'LayerNorm：每一行均值0']):
    ax.imshow(M, cmap='coolwarm'); ax.set_title(t, fontsize=10)
    for i in range(3):
        for j in range(4): ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('图5  BatchNorm 跨样本算统计量，LayerNorm 只在一个样本内部算')
fig.tight_layout(); fig.savefig('fig5_bn_vs_ln.png', dpi=150); plt.close(fig)

# ================= 汇总 =================
section('汇总')
print(f'瞎猜 3.2958 | bigram dev {bigram_dev:.4f}')
print(f'MLP naive 初始化   初始 {res["naive"]["l0"]:.3f} → dev {res["naive"]["dev"]:.4f}')
print(f'MLP Kaiming 初始化 初始 {res["kaiming"]["l0"]:.3f} → dev {res["kaiming"]["dev"]:.4f}（test {test_loss:.4f}）')
print('激活函数 dev：' + '  '.join(f'{k} {v[1]:.4f}' for k, v in act_res.items()))
print(f'W1 未缩放：无 BN dev {bn_res[False]["dev"]:.4f} | 有 BN dev {bn_res[True]["dev"]:.4f}')
print(f'总用时 {time.time() - T0:.0f}s')

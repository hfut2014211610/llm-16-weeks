import torch
import torch.nn.functional as F

words = open('names.txt').read().splitlines()
chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0                                  # '.' 同时表示开头和结尾
itos = {i: s for s, i in stoi.items()}

# --- 方法一：计数 ---
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    cs = ['.'] + list(w) + ['.']
    for a, b in zip(cs, cs[1:]):
        N[stoi[a], stoi[b]] += 1
P = (N + 1).float()                            # +1 平滑，避免概率为 0
P /= P.sum(1, keepdim=True)

nll, n = 0.0, 0
for w in words:
    cs = ['.'] + list(w) + ['.']
    for a, b in zip(cs, cs[1:]):
        nll -= torch.log(P[stoi[a], stoi[b]]); n += 1
print('计数法 平均NLL:', (nll / n).item())

# --- 方法二：神经网络（一层线性 + softmax）---
xs, ys = [], []
for w in words:
    cs = ['.'] + list(w) + ['.']
    for a, b in zip(cs, cs[1:]):
        xs.append(stoi[a]); ys.append(stoi[b])
xs, ys = torch.tensor(xs), torch.tensor(ys)
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
for k in range(100):
    logits = F.one_hot(xs, 27).float() @ W     # 等价于 W[xs]
    loss = F.cross_entropy(logits, ys) + 0.01 * (W ** 2).mean()
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad
print('神经网络 loss:', loss.item())

# --- 采样 ---
for _ in range(5):
    out, ix = [], 0
    while True:
        ix = torch.multinomial(P[ix], 1, generator=g).item()
        if ix == 0: break
        out.append(itos[ix])
    print(''.join(out))

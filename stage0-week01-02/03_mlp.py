import random
import torch
import torch.nn.functional as F

words = open('names.txt').read().splitlines()
chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}; stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
block_size = 3                                  # 用前 3 个字符预测下一个

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
Xtr, Ytr = build_dataset(words[:n1])            # 80% 训练
Xdev, Ydev = build_dataset(words[n1:n2])        # 10% 验证（调超参）
Xte, Yte = build_dataset(words[n2:])            # 10% 测试（最后才看）

g = torch.Generator().manual_seed(2147483647)
n_embd, n_hidden = 10, 200
C  = torch.randn((27, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3) / (n_embd * block_size) ** 0.5
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, 27), generator=g) * 0.01   # 输出层缩小 → 初始 loss≈3.29
b2 = torch.zeros(27)
params = [C, W1, b1, W2, b2]
for p in params: p.requires_grad = True

def forward(X):
    emb = C[X].view(X.shape[0], -1)             # (B, 3*n_embd)
    h = torch.tanh(emb @ W1 + b1)               # (B, n_hidden)
    return h @ W2 + b2                          # (B, 27)

steps = 50000
for i in range(steps):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)   # minibatch
    loss = F.cross_entropy(forward(Xtr[ix]), Ytr[ix])
    for p in params: p.grad = None
    loss.backward()
    lr = 0.1 if i < steps // 2 else 0.01        # 学习率衰减
    for p in params: p.data += -lr * p.grad
    if i % 10000 == 0: print(i, round(loss.item(), 4))

with torch.no_grad():
    print('train', F.cross_entropy(forward(Xtr), Ytr).item())
    print('dev  ', F.cross_entropy(forward(Xdev), Ydev).item())
    for _ in range(5):
        ctx, out = [0] * block_size, []
        while True:
            probs = F.softmax(forward(torch.tensor([ctx])), dim=1)
            ix = torch.multinomial(probs, 1, generator=g).item()
            ctx = ctx[1:] + [ix]
            if ix == 0: break
            out.append(itos[ix])
        print(''.join(out))

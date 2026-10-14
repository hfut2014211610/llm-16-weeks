import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- 超参（GPU 配置；CPU 上把注释里的小配置换进来）----
batch_size, block_size = 64, 256     # CPU: 32, 64
n_embd, n_head, n_layer = 384, 6, 6  # CPU: 64, 4, 4
dropout, lr = 0.2, 3e-4              # CPU: 0.0, 1e-3
max_iters, eval_interval, eval_iters = 5000, 500, 200
if len(sys.argv) > 1 and sys.argv[1] == 'cpu':
    batch_size, block_size, n_embd, n_head, n_layer, dropout, lr = 32, 64, 64, 4, 4, 0.0, 1e-3
    max_iters, eval_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3000, 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

text = open('input.txt', encoding='utf-8').read()
chars = sorted(set(text)); vocab_size = len(chars)          # 65
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])   # 目标 = 右移一位
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval(); out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split)); losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(); return out

class Head(nn.Module):
    """单头因果自注意力"""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)            # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5             # (B,T,T) 缩放点积
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # 因果掩码
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v                                                  # (B,T,hs)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    """通信（注意力）+ 计算（FFN），Pre-LN + 残差"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))               # (B,T,vocab)
        loss = None if targets is None else F.cross_entropy(
            logits.view(B * T, -1), targets.view(B * T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])                     # 截断到上下文长度
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

model = GPT().to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M 参数')
opt = torch.optim.AdamW(model.parameters(), lr=lr)
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        l = estimate_loss(); print(f"step {it}: train {l['train']:.4f}, val {l['val']:.4f}")
    _, loss = model(*get_batch('train'))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 300)[0].tolist()))

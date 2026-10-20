"""从第 1 批的 gpt.py 一步步改成 Llama。每个改动都能用命令行单独开关。

GPT 基线（等价于 gpt.py cpu）： python llama.py --steps 3000
完整 Llama 小模型：            python llama.py --steps 3000 --norm rms --pos rope --ffn swiglu --kv_heads 2 --bias 0
"""
import argparse, json, math, os, time, urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

p = argparse.ArgumentParser()
p.add_argument('--steps', type=int, default=3000)
p.add_argument('--norm', choices=['ln', 'rms'], default='ln')            # 第 2 章
p.add_argument('--pos', choices=['learned', 'rope'], default='learned')  # 第 3 章
p.add_argument('--rope_theta', type=float, default=10000.0)              # 第 3 章 rope base
p.add_argument('--ffn', choices=['relu', 'swiglu'], default='relu')      # 第 4 章
p.add_argument('--kv_heads', type=int, default=None)                     # 第 5 章，默认 = n_head（普通 MHA）
p.add_argument('--bias', type=int, default=1)                            # Llama 全部 Linear 无 bias
p.add_argument('--tie', type=int, default=0)                             # 输入/输出 embedding 是否共享
p.add_argument('--n_embd', type=int, default=64)
p.add_argument('--n_head', type=int, default=4)
p.add_argument('--n_layer', type=int, default=4)
p.add_argument('--block_size', type=int, default=64)
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--eval_interval', type=int, default=250)
p.add_argument('--eval_iters', type=int, default=50)
p.add_argument('--extrap', type=int, default=0, help='训练后在更长上下文上测 val loss（只对 rope 有意义）')
p.add_argument('--log', type=str, default='', help='把 loss 曲线写成 json')
p.add_argument('--sample', type=int, default=300)
p.add_argument('--seed', type=int, default=1337)
args = p.parse_args()
kv_heads = args.kv_heads or args.n_head
assert args.n_head % kv_heads == 0, 'n_head 必须是 kv_heads 的整数倍'
torch.manual_seed(args.seed)
torch.set_num_threads(max(1, os.cpu_count() or 1))
device = 'cpu'

# ---- 数据：tinyshakespeare，自动下载（不入库）----
if not os.path.exists('input.txt'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'input.txt')
text = open('input.txt', encoding='utf-8').read()
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

def get_batch(split, T=None):
    T = T or args.block_size
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - T, (args.batch_size,))
    x = torch.stack([d[i:i + T] for i in ix]); y = torch.stack([d[i + 1:i + T + 1] for i in ix])
    return x, y

# ---------------- 第 2 章：RMSNorm ----------------
class RMSNorm(nn.Module):
    """不减均值、没有 bias：x / sqrt(mean(x^2) + eps) * weight"""
    def __init__(self, dim, eps=1e-5):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def Norm(dim):
    return RMSNorm(dim) if args.norm == 'rms' else nn.LayerNorm(dim, bias=bool(args.bias))

# ---------------- 第 3 章：RoPE ----------------
def rope_cache(head_dim, T, theta):
    # 每两维一组，第 i 组的转速 = theta^(-2i/d)；位置 m 转 m*转速 弧度
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))   # (hd/2,)
    ang = torch.outer(torch.arange(T).float(), inv_freq)                              # (T, hd/2)
    return ang.cos(), ang.sin()

def apply_rope(x, cos, sin):
    # x: (B, H, T, hd)，把 (x0,x1),(x2,x3)... 每一对看成平面上的点，逆时针转 ang
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)

# ---------------- 第 5 章：注意力（MHA / GQA）----------------
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        C, H = args.n_embd, args.n_head
        self.H, self.KV, self.hd = H, kv_heads, C // H
        self.q = nn.Linear(C, H * self.hd, bias=False)
        self.k = nn.Linear(C, self.KV * self.hd, bias=False)     # GQA：K/V 只有 KV 个头
        self.v = nn.Linear(C, self.KV * self.hd, bias=False)
        self.proj = nn.Linear(H * self.hd, C, bias=bool(args.bias))
    def forward(self, x, rope):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.H, self.hd).transpose(1, 2)     # (B,H,T,hd)
        k = self.k(x).view(B, T, self.KV, self.hd).transpose(1, 2)    # (B,KV,T,hd)
        v = self.v(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        if rope is not None:                                         # RoPE 只转 Q 和 K，不动 V
            cos, sin = rope; q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        if self.KV != self.H:                                        # 每组 H/KV 个 Q 头共用一份 K/V
            k = k.repeat_interleave(self.H // self.KV, dim=1); v = v.repeat_interleave(self.H // self.KV, dim=1)
        att = (q @ k.transpose(-2, -1)) * self.hd ** -0.5
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        att = F.softmax(att.masked_fill(~mask, float('-inf')), dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)

# ---------------- 第 4 章：FFN（ReLU / SwiGLU）----------------
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        C, b = args.n_embd, bool(args.bias)
        if args.ffn == 'swiglu':
            hidden = int(8 * C / 3); hidden = 4 * math.ceil(hidden / 4)   # ≈ 8/3·d，凑成 4 的倍数：64→172
            self.w1 = nn.Linear(C, hidden, bias=b)   # gate
            self.w3 = nn.Linear(C, hidden, bias=b)   # up
            self.w2 = nn.Linear(hidden, C, bias=b)   # down
        else:
            self.fc = nn.Linear(C, 4 * C, bias=b); self.out = nn.Linear(4 * C, C, bias=b)
    def forward(self, x):
        if args.ffn == 'swiglu':
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.out(F.relu(self.fc(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.ffn = Norm(args.n_embd), Attention(), Norm(args.n_embd), FeedForward()
    def forward(self, x, rope):
        x = x + self.attn(self.ln1(x), rope)     # Pre-norm + 残差，GPT 和 Llama 一样
        return x + self.ffn(self.ln2(x))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        C = args.n_embd
        self.tok_emb = nn.Embedding(vocab_size, C)
        self.pos_emb = nn.Embedding(args.block_size, C) if args.pos == 'learned' else None
        self.blocks = nn.ModuleList([Block() for _ in range(args.n_layer)])
        self.ln_f = Norm(C)
        self.lm_head = nn.Linear(C, vocab_size, bias=bool(args.bias))
        if args.tie: self.lm_head.weight = self.tok_emb.weight
        maxT = max(args.block_size, args.extrap)
        if args.pos == 'rope':
            cos, sin = rope_cache(C // args.n_head, maxT, args.rope_theta)
            self.register_buffer('cos', cos, persistent=False); self.register_buffer('sin', sin, persistent=False)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None: x = x + self.pos_emb(torch.arange(T))
        rope = (self.cos, self.sin) if args.pos == 'rope' else None
        for blk in self.blocks: x = blk(x, rope)
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n):
        for _ in range(n):
            logits, _ = self(idx[:, -args.block_size:])
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits[:, -1], -1), 1)], dim=1)
        return idx

@torch.no_grad()
def estimate_loss(T=None):
    model.eval(); out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            _, loss = model(*get_batch(split, T)); losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(); return out

model = Model()
n_params = sum(p.numel() for p in model.parameters())
hd = args.n_embd // args.n_head
kv_bytes = 2 * args.n_layer * kv_heads * hd * 4      # K 和 V × 层数 × KV 头 × 头维 × fp32 4 字节
print(f"配置: norm={args.norm} pos={args.pos} ffn={args.ffn} heads={args.n_head} kv_heads={kv_heads} "
      f"bias={args.bias} tie={args.tie} rope_theta={args.rope_theta:g}")
print(f"参数量: {n_params} ({n_params / 1e6:.4f}M) | KV cache 每个 token: {kv_bytes} 字节 (fp32)")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
curve, t0 = [], time.time()
for it in range(args.steps + 1):
    if it % args.eval_interval == 0 or it == args.steps:
        l = estimate_loss(); curve.append((it, l['train'], l['val']))
        print(f"step {it}: train {l['train']:.4f}, val {l['val']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    if it == args.steps: break
    _, loss = model(*get_batch('train'))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
secs = time.time() - t0
res = dict(args=vars(args), params=n_params, kv_bytes_per_token=kv_bytes, curve=curve,
           final_train=curve[-1][1], final_val=curve[-1][2], seconds=round(secs, 1))
if args.extrap and args.pos == 'learned':
    print('学习型位置 embedding 只有 block_size 行，无法外推，跳过')
elif args.extrap:
    torch.manual_seed(0); l = estimate_loss(args.extrap)
    res['extrap'] = {'T': args.extrap, 'val': l['val']}
    print(f"外推：训练长度 {args.block_size}，在长度 {args.extrap} 上 val {l['val']:.4f}")
torch.manual_seed(42)
res['sample'] = decode(model.generate(torch.zeros((1, 1), dtype=torch.long), args.sample)[0].tolist())
print(res['sample'])
if args.log: json.dump(res, open(args.log, 'w'), ensure_ascii=False, indent=1)

"""讲义里所有回溯框的小例子（输出即讲义里的数字）。python snippets.py"""
import math, torch, torch.nn.functional as F
torch.set_printoptions(precision=4, sci_mode=False)

print("== 2 RMSNorm vs LayerNorm")
h = torch.tensor([2., 4., 6., 8.])
ms = (h ** 2).mean(); rms = ms.sqrt()
print('平方的均值', ms.item(), '均方根', round(rms.item(), 4))
print('RMSNorm  ', h / torch.sqrt(ms + 1e-5))
print('LayerNorm', F.layer_norm(h, (4,)))
h2 = h + 100
print('整体+100 后 RMSNorm', h2 / torch.sqrt((h2 ** 2).mean() + 1e-5))

print("== 3a 二维旋转")
def rot(x, y, deg):
    t = math.radians(deg); return (x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t))
print('cos/sin 30,60,90:', [(round(math.cos(math.radians(d)), 4), round(math.sin(math.radians(d)), 4)) for d in (30, 60, 90)])
print('(1,0) 转 90°', [round(v, 4) for v in rot(1, 0, 90)])
print('(2,1) 转 30°', [round(v, 4) for v in rot(2, 1, 30)], '长度', math.hypot(2, 1), math.hypot(*rot(2, 1, 30)))
z = complex(2, 1) * complex(math.cos(math.radians(30)), math.sin(math.radians(30)))
print('复数乘法同结果', round(z.real, 4), round(z.imag, 4))

print("== 3b 点积只看相对距离")
q, k = (1.0, 2.0), (3.0, -1.0)
step = 20   # 每个位置转 20°
for m, n in [(5, 3), (102, 100), (3, 5)]:
    qm, kn = rot(*q, m * step), rot(*k, n * step)
    print(f'q 在位置{m}, k 在位置{n}: 点积 {qm[0]*kn[0]+qm[1]*kn[1]:.4f}')

print("== 3c RoPE 每组转速")
hd, theta = 16, 10000.0
inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
print('head_dim=16 theta=1e4 每步转角(弧度):', inv)
print('最慢一组转一圈需要的位置数:', round(2 * math.pi / inv[-1].item()))
for th in (1e4, 5e5):
    inv128 = 1.0 / (th ** (torch.arange(0, 128, 2).float() / 128))
    print(f'head_dim=128 theta={th:g} 最慢一组一圈:', round(2 * math.pi / inv128[-1].item()))

print("== 4 SiLU 与门控")
x = torch.tensor([-2., 0., 2.])
print('relu', F.relu(x), 'sigmoid', torch.sigmoid(x), 'silu', F.silu(x))
gate, up = torch.tensor([-3., 0.5, 4.]), torch.tensor([2., 2., 2.])
print('silu(gate)', F.silu(gate), ' * up =', F.silu(gate) * up)
d = 64; hid = 4 * math.ceil(int(8 * d / 3) / 4)
print('ReLU FFN 参数(无bias)', 2 * d * 4 * d, ' SwiGLU 隐藏维', hid, '参数', 3 * d * hid)
d = 4096; print('Llama 原始 8/3 规则 d=4096 ->', int(8 * d / 3), '; Llama-3-8B 实际 intermediate_size=14336 (=3.5d)')

print("== 5 GQA repeat_interleave")
k = torch.arange(2).view(1, 2, 1, 1).float()   # 2 个 KV 头，内容分别是 0 和 1
print(k.repeat_interleave(2, dim=1).flatten())  # 4 个 Q 头看到的 K：0 0 1 1

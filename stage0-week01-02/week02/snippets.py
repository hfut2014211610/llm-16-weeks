# snippets.py —— 讲义回溯框里的小例子（输出即讲义里的数字）
import math, torch
import torch.nn.functional as F
print('1)', -math.log(1/27))
print('2)', F.cross_entropy(torch.zeros(1, 27), torch.tensor([5])).item())
lg = torch.tensor([[0., 0., 8.] + [0.]*24])
print('3) 自信错', F.cross_entropy(lg, torch.tensor([0])).item(), ' 自信对', F.cross_entropy(lg, torch.tensor([2])).item())
x = torch.tensor([0.5, 2.0, 5.0], requires_grad=True)
y = torch.tanh(x); y.sum().backward()
print('4) tanh', y.detach(), ' 导数 1-tanh^2', x.grad)
torch.manual_seed(0)
for n in [1, 30, 300]:
    s = (torch.randn(2000, n) @ torch.randn(n, 500)).std().item()
    s2 = (torch.randn(2000, n) @ (torch.randn(n, 500) / n**0.5)).std().item()
    print(f'5) n={n:3d} 不缩放 std {s:.2f}   除以√n 后 {s2:.2f}   √n={n**0.5:.2f}')
h = torch.tensor([2., 4., 6., 8.])
print('6) LN', F.layer_norm(h, (4,)))
print('7) RMSNorm', h / h.pow(2).mean().sqrt())

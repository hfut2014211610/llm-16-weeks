"""消融实验：同样步数、同样数据、同样种子，逐项把 GPT 换成 Llama。
用法： python ablation.py            # 默认 3000 步
      python ablation.py 2000       # 步数少一点更快
结果写到 results/*.json 和 results/table.md；已跑过的会跳过。
"""
import json, os, subprocess, sys

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
os.makedirs('results', exist_ok=True)
L = '--norm rms --pos rope --ffn swiglu --kv_heads 2'
RUNS = [  # (名字, 说明, 参数)
    ('A_gpt',      'GPT 基线（LayerNorm+学习型位置+ReLU+MHA+bias）', ''),
    ('B_rms',      '+RMSNorm',                       '--norm rms'),
    ('C_rope',     '+RoPE',                          '--norm rms --pos rope'),
    ('D_swiglu',   '+SwiGLU',                        '--norm rms --pos rope --ffn swiglu'),
    ('E_gqa',      '+GQA（4 个 Q 头 / 2 个 KV 头）',    L),
    ('F_llama',    '+去掉 bias = 完整 Llama 结构',     L + ' --bias 0'),
    ('A_gpt_s2',   'GPT 基线，换种子 2（看噪声）',       '--seed 2'),
    ('F_llama_s2', '完整 Llama，换种子 2（看噪声）',     L + ' --bias 0 --seed 2'),
    ('X_theta100', '可选：rope_theta=100',            L + ' --bias 0 --rope_theta 100'),
    ('X_mqa',      '可选：kv_heads=1（MQA）',          '--norm rms --pos rope --ffn swiglu --kv_heads 1 --bias 0'),
]
only = sys.argv[2:]   # 也可以只跑指定的几个：python ablation.py 3000 A_gpt B_rms
for name, desc, a in RUNS:
    if only and name not in only: continue
    out = f'results/{name}.json'
    if os.path.exists(out): continue
    extra = ' --extrap 128' if 'rope' in a else ''
    cmd = f'{sys.executable} llama.py --steps {steps} --log {out}{extra} {a}'
    print('>>', cmd, flush=True)
    with open(f'results/{name}.log', 'w') as f:
        subprocess.run(cmd.split(), stdout=f, stderr=subprocess.STDOUT, check=True)

rows = ['| 运行 | 改动 | 参数量 | train | val | KV cache/token | 用时 |', '|---|---|---|---|---|---|---|']
for name, desc, _ in RUNS:
    out = f'results/{name}.json'
    if not os.path.exists(out): continue
    r = json.load(open(out))
    rows.append(f"| {name} | {desc} | {r['params']:,} | {r['final_train']:.4f} | {r['final_val']:.4f} | "
                f"{r['kv_bytes_per_token']} B | {r['seconds']:.0f}s |")
open('results/table.md', 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))

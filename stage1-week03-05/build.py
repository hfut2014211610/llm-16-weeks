"""handout.md + results/*.json → PDF。所有 {{占位符}} 都从真实运行结果填入。 python build.py"""
import json, os, markdown, weasyprint

R = {f[:-5]: json.load(open('results/' + f)) for f in os.listdir('results') if f.endswith('.json')}
A, F = R['A_gpt'], R['F_llama']
names = [('A_gpt', 'A', 'GPT 基线（gpt.py 同构）'), ('B_rms', 'B', '+RMSNorm'), ('C_rope', 'C', '+RoPE（去掉位置表）'),
         ('D_swiglu', 'D', '+SwiGLU'), ('E_gqa', 'E', '+GQA（4 Q 头 / 2 KV 头）'), ('F_llama', 'F', '+去 bias = 完整 Llama')]
rows = ['| 运行 | 累积改动 | 参数量 | train | val | Δval vs A | KV cache / token（fp32） |', '|---|---|---|---|---|---|---|']
for k, tag, desc in names:
    r = R[k]
    rows.append(f"| {tag} | {desc} | {r['params']:,} | {r['final_train']:.4f} | {r['final_val']:.4f} | "
                f"{r['final_val'] - A['final_val']:+.4f} | {r['kv_bytes_per_token']} B |")
for k, desc in [('A_gpt_s2', 'A 换种子 2'), ('F_llama_s2', 'F 换种子 2')]:
    if k in R:
        r = R[k]; rows.append(f"| {desc} | 同上结构，只换随机种子 | {r['params']:,} | {r['final_train']:.4f} | "
                              f"{r['final_val']:.4f} | {r['final_val'] - A['final_val']:+.4f} | {r['kv_bytes_per_token']} B |")
table = '\n'.join(rows)

noise = []
if 'A_gpt_s2' in R: noise.append(abs(R['A_gpt_s2']['final_val'] - A['final_val']))
if 'F_llama_s2' in R: noise.append(abs(R['F_llama_s2']['final_val'] - F['final_val']))
noise_s = f"{max(noise):.3f}" if noise else '（未测）'

vals = {k: R[k]['final_val'] for k, _, _ in names}
steps = []
prev = 'A_gpt'
for k, tag, desc in names[1:]:
    steps.append(f"{desc.split('（')[0]} {vals[k] - vals[prev]:+.4f}"); prev = k
read = (f"- 逐项相对上一行的 val 变化：{'；'.join(steps)}。\n"
        f"- 完整 Llama（F）对 GPT 基线（A）：val {F['final_val']:.4f} vs {A['final_val']:.4f}，"
        f"差 {F['final_val'] - A['final_val']:+.4f}；参数量 {F['params']:,} vs {A['params']:,}"
        f"（少 {A['params'] - F['params']:,}，主要来自删掉 64×64 的位置表、减少 K/V 投影和去 bias）。\n"
        f"- 两个种子平均：A {(A['final_val'] + R['A_gpt_s2']['final_val']) / 2:.4f}，F {(F['final_val'] + R['F_llama_s2']['final_val']) / 2:.4f}，"
        f"差 {(F['final_val'] + R['F_llama_s2']['final_val'] - A['final_val'] - R['A_gpt_s2']['final_val']) / 2:+.4f}，比单种子的差距更可信。\n"
        f"- KV cache：GQA 把每 token 缓存从 {A['kv_bytes_per_token']} 字节减半到 {F['kv_bytes_per_token']} 字节，这一项在任何规模都成立。\n"
        f"- 耗时：A {A['seconds']:.0f}s，F {F['seconds']:.0f}s（CPU 小模型上几乎一样；Llama 结构的速度收益在大模型推理时读写 KV cache 上，这里体现不出）。")
diff = F['final_val'] - A['final_val']
phrase = (f"val 比 GPT 低 {-diff:.3f}" if diff < 0 else f"val 比 GPT 高 {diff:.3f}")
ext = R['C_rope'].get('extrap', {})
ext_s = f"为 {ext['val']:.4f}（训练长度 64 上是 {R['C_rope']['final_val']:.4f}）" if ext else '（未测）'
if 'F_llama' in R and 'extrap' in F:
    ext_s = f"为 {F['extrap']['val']:.4f}（完整 Llama；训练长度 64 上是 {F['final_val']:.4f}）"
total_min = sum(r['seconds'] for r in R.values()) / 60
sub = dict(EXT_C=f"{R['C_rope']['extrap']['val']:.4f}", EXT_F=f"{F['extrap']['val']:.4f}", STEPS=str(A['args']['steps']), ABL_TABLE=table, ABL_READ=read, NOISE=noise_s,
           SAMPLE=F['sample'].strip(), SUMMARY_PHRASE=phrase, EXTRAP=ext_s, F_VAL=f"{F['final_val']:.4f}",
           TOTAL_MIN=f"{total_min:.0f}",
           THETA100=f"{R['X_theta100']['final_val']:.4f}" if 'X_theta100' in R else '（未测）',
           MQA_PARAMS=f"{R['X_mqa']['params']:,}" if 'X_mqa' in R else '（未测）',
           MQA_VAL=f"{R['X_mqa']['final_val']:.4f}" if 'X_mqa' in R else '（未测）',
           MQA_KV=str(R['X_mqa']['kv_bytes_per_token']) if 'X_mqa' in R else '（未测）')
md = open('handout.md', encoding='utf-8').read()
for k, v in sub.items(): md = md.replace('{{' + k + '}}', v)
assert '{{' not in md, md[md.index('{{'):md.index('{{') + 40]
open('handout.filled.md', 'w', encoding='utf-8').write(md)

html = markdown.markdown(md, extensions=['fenced_code', 'tables', 'md_in_html', 'codehilite'],
                         extension_configs={'codehilite': {'noclasses': True, 'pygments_style': 'friendly'}})
css = """
@page { size: A4; margin: 17mm 15mm; @bottom-center { content: counter(page); font-size: 9pt; color:#888; } }
body { font-family: 'Noto Sans SC', sans-serif; font-size: 10.3pt; line-height: 1.6; color:#222; }
h1 { font-size: 20pt; margin: 0 0 2pt; color:#1a3c6e; }
.sub { color:#666; margin-top:0; }
h2 { font-size: 14pt; color:#1a3c6e; border-bottom: 1.5px solid #c9d6ea; padding-bottom:3pt; margin-top: 16pt; page-break-after: avoid; }
hr { border:none; margin: 4pt 0; }
pre, code { font-family: 'DejaVu Sans Mono', 'Noto Sans SC', monospace; font-size: 8.5pt; }
pre { background:#f6f8fa; padding:6pt 8pt; border-radius:4px; white-space: pre-wrap; word-break: break-all; page-break-inside: avoid; }
.codehilite { background:#f6f8fa !important; border-radius:4px; }
p code, li code, td code { background:#eef1f5; padding:0 3px; border-radius:3px; }
table { border-collapse: collapse; width:100%; margin: 6pt 0; font-size: 8.9pt; page-break-inside: avoid; }
th, td { border:1px solid #d0d7de; padding:3pt 5pt; text-align:left; vertical-align: top; }
th { background:#eef3fa; }
img { max-width: 100%; display:block; margin: 6pt auto; page-break-inside: avoid; }
p:has(img) { page-break-inside: avoid; }
.back { background:#fff8e6; border-left: 4px solid #e0a800; padding: 3pt 10pt; margin: 8pt 0; border-radius: 3px; }
.back pre, .back .codehilite { background:#fffdf5 !important; }
.foot { color:#777; font-size:9pt; margin-top:14pt; }
"""
doc = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>"
weasyprint.HTML(string=doc, base_url='.').write_pdf('第2批讲义-从GPT到Llama.pdf')
print(table); print(read); print('ok')

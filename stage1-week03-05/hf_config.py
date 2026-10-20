"""把 llama.py 的命令行参数对应到 Hugging Face config.json / GGUF 字段，并计算 KV cache。
不联网、不下载模型；真实模型的数值按官方 config.json 抄写（以官方仓库为准）。 python hf_config.py"""

# 我们的小模型（llama.py 完整 Llama 配置）写成 HF 风格
ours = dict(hidden_size=64, intermediate_size=172, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, rope_theta=10000.0, rms_norm_eps=1e-5, max_position_embeddings=64,
            vocab_size=65, tie_word_embeddings=False, hidden_act='silu', attention_bias=False)
llama3_8b = dict(hidden_size=4096, intermediate_size=14336, num_hidden_layers=32, num_attention_heads=32,
                 num_key_value_heads=8, rope_theta=500000.0, rms_norm_eps=1e-5, max_position_embeddings=8192,
                 vocab_size=128256, tie_word_embeddings=False, hidden_act='silu', attention_bias=False)
qwen25_7b = dict(hidden_size=3584, intermediate_size=18944, num_hidden_layers=28, num_attention_heads=28,
                 num_key_value_heads=4, rope_theta=1000000.0, rms_norm_eps=1e-6, max_position_embeddings=32768,
                 vocab_size=152064, tie_word_embeddings=False, hidden_act='silu', attention_bias='True(仅QKV)')

mapping = [  # HF 字段, llama.py 参数, GGUF 键 / llama.cpp 日志名
    ('hidden_size', '--n_embd', 'llama.embedding_length / n_embd'),
    ('intermediate_size', '≈8/3·n_embd (SwiGLU)', 'llama.feed_forward_length / n_ff'),
    ('num_hidden_layers', '--n_layer', 'llama.block_count / n_layer'),
    ('num_attention_heads', '--n_head', 'llama.attention.head_count / n_head'),
    ('num_key_value_heads', '--kv_heads', 'llama.attention.head_count_kv / n_head_kv'),
    ('rope_theta', '--rope_theta', 'llama.rope.freq_base / freq_base'),
    ('rms_norm_eps', 'RMSNorm eps=1e-5', 'llama.attention.layer_norm_rms_epsilon / f_norm_rms_eps'),
    ('max_position_embeddings', '--block_size', 'llama.context_length / n_ctx_train'),
    ('vocab_size', '65（字符）', 'tokenizer 词表大小 / n_vocab'),
    ('tie_word_embeddings', '--tie', '有无独立 output.weight 张量'),
    ('hidden_act', "--ffn swiglu → 'silu'", '（架构固定）'),
    ('attention_bias', '--bias', '有无 attn_q.bias 等张量'),
]
print(f"{'HF 字段':26s}{'ours':>12s}{'Llama-3-8B':>14s}{'Qwen2.5-7B':>16s}   llama.py 参数 | GGUF/llama.cpp")
for f, a, g in mapping:
    print(f"{f:26s}{str(ours[f]):>12s}{str(llama3_8b[f]):>14s}{str(qwen25_7b[f]):>16s}   {a} | {g}")

def kv_cache(cfg, ctx, bytes_per=2, kv=None):
    kv = kv or cfg['num_key_value_heads']; hd = cfg['hidden_size'] // cfg['num_attention_heads']
    per_tok = 2 * cfg['num_hidden_layers'] * kv * hd * bytes_per        # K 和 V
    return per_tok, per_tok * ctx

print('\n== KV cache（fp16，2 字节）')
for name, cfg in [('Llama-3-8B', llama3_8b), ('Qwen2.5-7B', qwen25_7b)]:
    for label, kv in [('GQA 实际', None), ('假如 MHA', cfg['num_attention_heads']), ('假如 MQA', 1)]:
        per, tot = kv_cache(cfg, 8192, kv=kv)
        print(f"{name:11s} {label:7s} kv_heads={kv or cfg['num_key_value_heads']:>2}: 每 token {per/1024:.0f} KiB, "
              f"8192 上下文 {tot/2**30:.3f} GiB, 32 并发×8192 {32*tot/2**30:.1f} GiB")
per, tot = kv_cache(llama3_8b, 8192, bytes_per=1)
print(f"Llama-3-8B q8_0 缓存(≈1 字节/数，忽略缩放因子): 8192 上下文约 {tot/2**30:.3f} GiB")
per, tot = kv_cache(llama3_8b, 131072)
print(f"Llama-3-8B fp16 若开到 128K 上下文: {tot/2**30:.1f} GiB")

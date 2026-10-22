# 阶段 1 · 第 3 周：从 GPT 到 Llama（第 2 批）

讲义：`第2批讲义-从GPT到Llama.pdf`（源文件 `讲义-第2批-从GPT到Llama.md`）

```bash
python llama.py --steps 3000 --norm rms --pos rope --ffn swiglu --kv_heads 2   # 完整 Llama 结构
python ablation.py 3000                                                         # 逐项替换的消融，结果写入 results/
python hf_config.py                                                             # 对照 HF config 字段与 KV cache 计算
```

数据 `input.txt`（tinyshakespeare）首次运行自动下载，不入库。消融真实结果见 `results/table.md`：CPU 3000 步，GPT 基线 val 1.7755，完整 Llama 1.6901，主要收益来自 RoPE。

深挖章节 A 已加入 RoPE 原文配读：苏剑林《Transformer升级之路：2、博采众长的旋转式位置编码》（https://kexue.fm/archives/8265），在讲义末尾，逐节标了对应章节和可跳过的推导。

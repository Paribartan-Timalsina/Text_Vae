# Encoder × decoder combos

Four backbone pairings for the LangVAE Table-1 comparison. Each file is a thin
**delta** layered over `base.yaml` + `default.yaml` — it only overrides the
backbone names and the per-combo knobs (KV fan-out, dtype, batch size).

| combo | encoder | decoder | cost | fan-out | kv_proj |
|-------|---------|---------|------|---------|---------|
| `bert_gpt2` | bert-base-cased | gpt2 | cheap | 50 | ~119M |
| `flant5_gpt2` | google/flan-t5-base | gpt2 | cheap | 50 | ~119M |
| `bert_llama` | bert-base-cased | meta-llama/Llama-3.2-3B | A100 | 50 | ~370M |
| `bert_mistral` | bert-base-cased | mistralai/Mistral-7B-v0.3 | A100-80GB | 50 | ~423M |

All four use **full KV fan-out** (`kv_fanout_len=50`), matching LangVAE's per-layer
W_m, so the comparison is fair across decoders.

## Run

Train / eval by layering three configs (base → default → combo):

```bash
python -m src.pipelines.train_vae \
    --config configs/base.yaml configs/vae/default.yaml configs/vae/combos/bert_llama.yaml
```

(`--config` takes all paths in one flag, merged left→right. Same for `eval_vae`.)

## Gated models

`Llama-3.2-3B` and `Mistral-7B-v0.3` are **gated** on Hugging Face. Before running
those combos:

1. Accept the model license on its HF page.
2. `huggingface-cli login` (or set the `HF_TOKEN` env var).

`bert-base-cased`, `google/flan-t5-base`, and `gpt2` are open — no login needed.

## KV fan-out cost across decoders

`kv_proj` = `latent_dim × (n_layer × 2 × n_kv_head × head_dim × kv_fanout_len)`.
Because Llama/Mistral use **grouped-query attention** (few KV heads: 8), the fan-out
stays small even on big models: ~119M (GPT-2), ~370M (Llama-3B), ~423M (Mistral-7B)
at `kv_fanout_len=50`. That's the same order as LangVAE's own per-layer W_m (~428M),
so we keep full fan-out on every combo for a fair, matched comparison. All are
trainable on an A100 (the decoder backbone stays frozen; only `kv_proj` + LoRA +
Perceiver train).

## Memory / quantization

The big-decoder combos default to **bf16, no 4-bit** — a 3B/7B model in bf16 fits
on an A100, and bf16 works with the training pipeline's `model.to(device)`.

If you must run a big decoder on a smaller GPU (e.g. T4 16 GB), set
`decoder.load_in_4bit: true` (QLoRA). **Caveat:** a 4-bit model cannot be moved by
`model.to(device)`, which the train/eval pipelines currently call — so 4-bit needs
`device_map` placement and a pipeline tweak to skip `.to(device)` for the quantized
LM. That tweak isn't wired yet; open an issue / ask before using 4-bit.

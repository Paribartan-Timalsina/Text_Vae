# Standalone SequenceVAE pipeline

A self-contained, end-to-end **SequenceVAE** project extracted from
`Latent-Text-Diffusion-Model`. It contains only the VAE — no diffusion model,
no null classifier, no LangVAE. The code, structure, and config layout mirror
the original repo; the diffusion/LangVAE pieces and their config sections have
been removed.

The pipeline is: **train → export latents (+ quality gate) → inference**.

## Layout

```
vae/
├── configs/
│   ├── base.yaml          # seed, paths, encoder
│   └── vae/default.yaml   # vae_arch, vae_training, quality_gate
├── src/
│   ├── config/            # schema, loader, validation, seed
│   ├── data/              # tokenization, SQuAD + EntailmentBank datasets, sampler, loaders
│   ├── models/vae/        # SequenceVAE: encoder, decoder, output head, loss, reparameterize, decoding
│   ├── pipelines/         # train_vae, export_latents, quality_gate
│   ├── training/          # optimizer, checkpoint, ema, grad_utils
│   ├── evaluation/        # squad_metrics, text_metrics, normalize, latent_analysis
│   └── utils/             # device, logging, pretrained_embeddings
├── tests/
└── pyproject.toml
```

## Setup

```bash
cd vae
pip install -e .          # or: uv pip install -e .
```

All commands are run **from inside the `vae/` directory** (it is the project
root, so the `from src.X import ...` imports resolve).

## Step 1 — Train the SequenceVAE

```bash
python -m src.pipelines.train_vae \
    --config configs/base.yaml configs/vae/default.yaml
```

Trains the encoder + decoder from scratch on SQuAD v2 answer spans (set
`vae_training.dataset: entailment_bank` to train on EntailmentBank instead).
Best checkpoint is saved to `checkpoints/vae_best.pt`.

Any config value can be overridden on the CLI with dot notation, e.g.:

```bash
python -m src.pipelines.train_vae \
    --config configs/base.yaml configs/vae/default.yaml \
    --vae_training.epochs 2 --vae_training.batch_size 8
```

## Step 2 — Export latents + quality gate

```bash
python -m src.pipelines.export_latents \
    --config configs/base.yaml configs/vae/default.yaml \
    --vae_checkpoint checkpoints/vae_best.pt
```

Encodes the full dataset deterministically (μ only), writes
`latents/latent_dataset_{train,val}.pt` and `latents/normalization_stats.pt`,
and runs the quality gate (reconstruction accuracy, active latent dims,
ans/no-ans centroid distance). Raises `RuntimeError` if the gate fails.

## Inference (encode → decode)

```python
import torch
from src.config.loader import load_config
from src.data.tokenization import create_tokenizer
from src.models.vae.vae import SequenceVAE
from src.training.checkpoint import load_checkpoint

cfg = load_config(["configs/base.yaml", "configs/vae/default.yaml"])
tokenizer = create_tokenizer(cfg.encoder.model_name)

ckpt = load_checkpoint("checkpoints/vae_best.pt")
placeholder = torch.zeros(len(tokenizer), cfg.vae_arch.embed_dim)
vae = SequenceVAE(cfg.vae_arch, pretrained_embeddings=placeholder)
vae.load_state_dict(ckpt["model_state_dict"])
vae.eval()

batch = tokenizer("the eiffel tower", return_tensors="pt")
_, mu, _ = vae.encode(batch["input_ids"], batch["attention_mask"], deterministic=True)
token_ids = vae.decode_to_tokens(mu, strategy="greedy", max_len=50)
print(tokenizer.decode(token_ids[0], skip_special_tokens=True))
```

## Tests

```bash
pytest tests/ -q
```

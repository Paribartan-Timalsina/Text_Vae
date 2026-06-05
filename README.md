# Standalone Sequence VAE

A self-contained copy of **only** the VAE part of the parent project — model,
loss, training loop, data loading, and the metrics needed to train and validate
the VAE on SQuAD v2. All diffusion / null-classifier / sampler / export code has
been removed so this folder is small and easy to iterate on.

This is a **copy**, not a move: the original code under the repo's top-level
`src/` is untouched. The file layout here mirrors that structure one-to-one
(one file per component) so anything you learn here maps back directly.

## Layout

```
vae_standalone/
├── main.py                       # entry point: parse config → train_vae
├── configs/
│   └── vae.yaml                  # base + vae config merged into one file
└── src/
    ├── config/                   # schema (VAE sections only), loader, validation, seed
    ├── data/                     # tokenization, SQuAD dataset, loaders, sampler
    ├── models/
    │   ├── positional.py         # sinusoidal positional encoding
    │   └── vae/                  # encoder, decoder, output_head, reparameterize, loss, vae
    ├── training/                 # ema, optimizer, grad_utils, checkpoint
    ├── evaluation/               # normalize, squad_metrics (EM/F1)
    ├── utils/                    # logging (wandb), pretrained_embeddings
    └── pipelines/
        └── train_vae.py          # the training loop
```

## Run

From **inside this folder** (it puts the local `src/` first on the path):

```bash
python main.py --config configs/vae.yaml
```

Override any config field with dot notation:

```bash
# turn the bag-of-words loss off once true_kl is healthy
python main.py --config configs/vae.yaml --vae_training.bow_loss_weight 0

# shrink the latent further
python main.py --config configs/vae.yaml --vae_arch.num_latent_tokens 4
```

Requires the same dependencies as the parent project (`torch`, `transformers`,
`datasets`, `pyyaml`, and optionally `wandb`). The parent repo's virtualenv works:

```bash
../.venv/bin/python main.py --config configs/vae.yaml
```

## Anti-collapse features (enabled in `configs/vae.yaml`)

The decoder is autoregressive, which makes text VAEs prone to **posterior
collapse** (the decoder reconstructs from teacher-forced tokens and ignores `z`;
generation then degenerates to `the the and`). The defaults here counter that:

| Feature | Config key | What it does |
|---|---|---|
| Per-position latent injection | `vae_arch.latent_pos_inject` | adds a K-pooled projection of `z` to **every** decoder token input, so `z` can't be bypassed |
| Bag-of-words auxiliary loss | `vae_arch.use_bow_head` + `vae_training.bow_loss_weight` | forces `z` to encode the answer's token set, independent of the decoder |
| Word dropout | `vae_training.word_dropout` | randomly masks teacher-forced inputs so the decoder must read `z` |
| Smaller latent | `vae_arch.num_latent_tokens` | 8×128 = 1024 dims (was 2048) — over-capacity collapses easily |
| Cyclical β | `vae_training.beta_schedule` | periodic low-KL windows to (re)learn to use `z` |
| No weight decay on variational heads | (in `train_vae.py`) | stops weight decay from driving μ→0 |

**Health signal:** watch `train/true_kl` (the unclamped KL), not `train/kl`
(which pins at `free_bits × K × D` once collapsed and is misleading).

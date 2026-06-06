"""Overfit smoke test for the rewritten frozen-backbone VAE.

Confirms the architecture fix: a handful of answers should be reconstructed
fluently from the latent z after a few hundred steps, and a wrong (rolled) z
should produce DIFFERENT text (f1_gap > 0). Uses real bert-base + gpt2 on CPU.
"""
import dataclasses as dc
import torch

from src.config.schema import Config
from src.data.tokenization import create_tokenizer, create_decoder_tokenizer
from src.data.squad_dataset import _tokenize_and_pad, make_decoder_target
from src.models.vae.vae import SequenceVAE

torch.manual_seed(0)
device = torch.device("cpu")

L = 20
cfg = Config()
cfg = dc.replace(
    cfg,
    vae_arch=dc.replace(cfg.vae_arch, latent_dim=64, num_latent_tokens=4, max_answer_len=L, num_heads=4),
    decoder=dc.replace(cfg.decoder, model_name="gpt2", max_answer_len=L, latent_pos_inject=True),
    encoder=dc.replace(cfg.encoder, model_name="bert-base-uncased"),
)

enc_tok = create_tokenizer(cfg.encoder.model_name)
dec_tok = create_decoder_tokenizer(cfg.decoder.model_name)

texts = [
    "the cat sat on the mat",
    "paris is the capital of france",
    "water boils at one hundred degrees",
    "the sun rises in the east",
    "dogs are loyal animals",
    "the earth orbits the sun",
]
enc_ids = torch.stack([_tokenize_and_pad(enc_tok, t, L)[0] for t in texts])
enc_mask = torch.stack([_tokenize_and_pad(enc_tok, t, L)[1] for t in texts])
dec_pairs = [make_decoder_target(dec_tok, t, L) for t in texts]
dec_ids = torch.stack([p[0] for p in dec_pairs])
dec_mask = torch.stack([p[1] for p in dec_pairs])

vae = SequenceVAE(cfg, encoder_vocab_size=len(enc_tok)).to(device)
n_train = sum(p.numel() for p in vae.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in vae.parameters())
print(f"trainable params: {n_train:,} / total {n_total:,}  ({100*n_train/n_total:.2f}%)")

opt = torch.optim.AdamW([p for p in vae.parameters() if p.requires_grad], lr=1e-3)

vae.train()
for step in range(201):
    _, z, mu, lv, ld = vae(enc_ids, enc_mask, dec_ids, dec_mask, beta=0.0)
    opt.zero_grad(); ld["total"].backward(); opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  recon={ld['recon'].item():.3f}  kl={ld['kl'].item():.3f}")

vae.eval()
with torch.no_grad():
    _, mu, _ = vae.encode(enc_ids, enc_mask, deterministic=True)
    gen = vae.decode_to_tokens(mu, strategy="greedy", max_len=L, eos_token_id=dec_tok.eos_token_id)
    mu_shuf = torch.roll(mu, 1, 0)
    gen_shuf = vae.decode_to_tokens(mu_shuf, strategy="greedy", max_len=L, eos_token_id=dec_tok.eos_token_id)

print("\n--- reconstruction from correct z ---")
for t, g in zip(texts, gen):
    print(f"  gold={t!r}\n  pred={dec_tok.decode(g.tolist(), skip_special_tokens=True).strip()!r}")
print("\n--- reconstruction from WRONG (rolled) z  [should differ → f1_gap>0] ---")
for t, g in zip(texts, gen_shuf):
    print(f"  gold={t!r}  pred={dec_tok.decode(g.tolist(), skip_special_tokens=True).strip()!r}")

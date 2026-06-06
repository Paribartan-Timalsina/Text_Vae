"""Frozen dataclass definitions for all configuration sections."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Optional


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths for data, checkpoints, and outputs."""

    data_dir: str = "data"  # Root data directory
    checkpoint_dir: str = "checkpoints"  # Where to save model checkpoints
    latent_dir: str = "latents"  # Precomputed latent vectors
    output_dir: str = "outputs"  # Generation outputs and logs


@dataclass(frozen=True)
class EncoderConfig:
    """Pretrained encoder settings."""

    model_name: str = "bert-base-uncased"  # HuggingFace model identifier
    hidden_dim: int = 768  # Encoder hidden dimension
    max_context_len: int = 384  # Max context token length
    max_question_len: int = 64  # Max question token length
    unfreeze_top_n: int = 0  # Number of top layers to unfreeze


@dataclass(frozen=True)
class DecoderConfig:
    """Frozen pretrained causal-LM decoder settings (LangVAE-style backbone).

    The decoder is a pretrained ``AutoModelForCausalLM`` (e.g. GPT-2) whose
    weights are FROZEN by default — only the latent-injection projections (and,
    if ``use_lora``, small LoRA adapters) are trained. This is the architectural
    fix for the from-scratch-decoder gibberish: a pretrained LM already produces
    fluent text, so the VAE only has to learn to read/write the latent.
    """

    model_name: str = "gpt2"  # HuggingFace causal-LM id used as the generator.
    max_answer_len: int = 50  # Max decoder tokens (reconstruction length).
    latent_pos_inject: bool = True  # Add a K-pooled projection of z to EVERY
    # decoder token embedding (not just the K-token prefix). Keeps z reachable at
    # every step so the frozen decoder cannot ignore it.
    use_lora: bool = False  # Attach LoRA adapters to the decoder (the "beat
    # LangVAE" lever — relaxes its frozen-decoder convergence limitation at <1%
    # trainable params). OFF by default; the architecture fix does not need it.
    lora_r: int = 16  # LoRA rank.
    lora_alpha: int = 32  # LoRA scaling.
    lora_dropout: float = 0.05  # LoRA dropout.


@dataclass(frozen=True)
class VAEArchConfig:
    """VAE architecture hyperparameters."""

    latent_dim: int = 128  # Latent space dimensionality
    embed_dim: int = 768  # Internal embedding dimension
    num_layers: int = 4  # Transformer layers in the ENCODER (and decoder when
    # ``decoder_num_layers`` is None)
    decoder_num_layers: Optional[int] = None  # Transformer layers in the DECODER.
    # None falls back to ``num_layers``. Set BELOW num_layers (e.g. encoder 4 /
    # decoder 2) to deliberately weaken the autoregressive decoder so it cannot
    # model p(answer) from teacher-forced tokens alone and is forced to read z —
    # the cure for powerful-decoder latent bypass (high free_bits-propped KL but
    # generation collapses to NULL; val EM/F1 pinned at the null rate).
    num_heads: int = 8  # Attention heads
    dropout: float = 0.1  # Dropout rate
    max_answer_len: int = 50  # Max answer token length
    num_latent_tokens: int = 4  # Pseudo-tokens for latent KV injection
    latent_pos_inject: bool = False  # Add a K-pooled projection of z to every
    # decoder token input (not just the KV prefix). Stops the causal decoder
    # from bypassing z via teacher forcing — the main posterior-collapse cure.
    use_bow_head: bool = False  # Build a bag-of-words head that predicts the
    # answer's token set from z alone (Zhao et al. 2017). Trained via
    # ``vae_training.bow_loss_weight``; forces z to stay informative.


@dataclass(frozen=True)
class VAETrainingConfig:
    """VAE training hyperparameters."""

    dataset: str = "squad_v2"  # VAE training corpus selector. "squad_v2" =
    # reconstruct SQuAD v2 answer texts (short, with NULLs). "entailment_bank" =
    # reconstruct EntailmentBank explanatory sentences (full declarative sentences,
    # no NULLs), matching the LangVAE paper (arXiv:2505.00004): all explanatory
    # ("cot") sentences, deduped, 99/1 split, one sentence per example.
    learning_rate: float = 5e-4  # Peak learning rate (1e-4 was ~10x too low →
    # recon stuck ~20; 5e-4 lets the decoder learn to read the latent)
    batch_size: int = 64  # Training batch size
    epochs: int = 30  # Maximum training epochs
    patience: int = 5  # Early stopping patience (val checks)
    warmup_steps: int = 500  # LR scheduler warmup
    weight_decay: float = 0.01  # AdamW weight decay
    grad_clip_max_norm: float = 5.0  # Gradient clipping threshold (1.0 throttled
    # the ~25M-param model's effective LR ~15-20x → underfit; see vae/default.yaml)
    grad_accum_steps: int = 1  # Gradient accumulation steps
    beta_start: float = 0.01  # KL weight at start
    beta_end: float = 0.5  # KL weight at end. 1.0 crushed the decoder at the
    # cyclical β peaks — val EM/F1 dipped to 0 exactly when β hit 1.0. Cap at 0.5.
    beta_warmup_steps: int = 10000  # Steps to ramp beta
    beta_schedule: str = "cyclical"  # "monotonic" or "cyclical"
    beta_cycles: int = 40  # Number of cycles (only used with "cyclical")
    target_kl: Optional[float] = (
        None  # KL hinge target (None = disabled). With K*D≈512 latent dims a
        # finite value like 20.0 drives the posterior toward ~0.02 nats/dim,
        # i.e. near-collapse. Rely on cyclical annealing + free_bits instead.
    )
    beta_cycle_ratio: float = 0.5  # Fraction of cycle spent ramping
    free_bits: float = 0.0625  # Per-dim KL ALLOWANCE the encoder may use penalty-
    # free (target-rate VAE). 0.02 collapsed (KL pinned at floor before the decoder
    # learned to read z); 0.3 over-corrected — the floor (K*D*0.3 ≈ 307 over 1024
    # dims) PROPPED train/kl to ~308 artificially while the powerful decoder bypassed
    # z entirely (gen → NULL, val EM/F1 stuck at the null rate, true_kl decaying).
    # 0.0625 over K*D=512 (num_latent_tokens 8→4) floors KL at ~32 nats — a real
    # target rate, not a prop — now safe because the decoder is also weakened
    # (decoder_num_layers). Watch train/true_kl AND val EM/F1: KL alive ≠ z used.
    ema_decay: float = 0.999  # EMA decay rate for validation weights
    val_every_n_steps: int = 500  # Validation frequency (steps)
    noise_aug_sigma: float = 0.0  # Extra Gaussian noise std added to z before
    # decode (decoder noise robustness for diffusion-time latents)
    noise_aug_prob: float = 0.0  # Per-step probability of applying noise aug
    null_train_fraction: float = 0.10  # Target fraction of unanswerable (NULL)
    # examples in the VAE *training* set. SQuAD v2 is ~33% null and the old
    # balanced sampler inflated that to 50%, starving real-answer reconstruction.
    # Subsampling nulls to 10% concentrates gradient on answer text. Only affects
    # VAE training; export/classifier/diffusion still see the full null set.
    null_loss_weight: float = 0.1  # Per-sample reconstruction-loss weight applied
    # to NULL examples (answerable examples keep weight 1.0). Further rebalances
    # the decoder's gradient toward answer text without removing nulls — they
    # still pass through the encoder so their latents stay structured for export.
    word_dropout: float = 0.1  # Probability of replacing each teacher-forced
    # decoder INPUT token with [MASK] during training (Bowman et al. 2016).
    # free_bits (=0.3) is the actual collapse cure — isolation showed free_bits
    # alone recovers reconstruction (acc 0.90) while word_dropout=0.5 HURT it
    # (acc 0.38, slower convergence). Kept small (0.1) as mild robustness for the
    # imperfect diffusion-sampled latents at inference. 0.0 disables.
    bow_loss_weight: float = 0.0  # Weight on the bag-of-words auxiliary loss
    # (requires vae_arch.use_bow_head). Added directly to the total loss like a
    # second reconstruction term; uses the same per-sequence-sum reduction so
    # the weight is comparable to recon. 0.0 disables. ~0.3-1.0 is typical.
    zforce_weight: float = 0.0  # Weight on the z-forcing auxiliary pass (Goyal
    # et al. 2017). A SECOND teacher-forced decode of the SAME decoder run with
    # word_dropout=1.0 — every input token is [MASK], so the decoder must
    # reconstruct from z (+ position) ALONE. Trains the deployed decoder under
    # generation-like conditions (no gold prev-token bypass) while the primary
    # word_dropout=word_dropout pass preserves inter-token fluency. The direct
    # cure for the teacher-forced/generation gap (good train recon, EM/F1 ~0).
    # Added to total like a second recon term (same reduction), so the weight is
    # comparable to recon. 0.0 disables. Requires a mask token id (BERT [MASK]).


@dataclass(frozen=True)
class QualityGateConfig:
    """Thresholds for latent quality gate checks."""

    min_recon_accuracy: float = 0.85  # Minimum token reconstruction accuracy
    min_mean_kl: float = 0.1  # Minimum mean KL divergence
    min_active_dims: int = 10  # Minimum active latent dimensions
    min_centroid_distance: float = 0.5  # Min L2 distance between ans/no-ans centroids
    active_dim_variance_threshold: float = 0.1  # Variance threshold for "active" dim
    max_dead_slots: int = 0  # Max collapsed latent slots (per-slot zero active dims)
    min_active_in_any_slot: int = 1  # Min active dims required in the weakest slot


@dataclass(frozen=True)
class Config:
    """Top-level configuration combining all sections."""

    seed: int = 42  # Global random seed
    paths: PathConfig = field(default_factory=PathConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    vae_arch: VAEArchConfig = field(default_factory=VAEArchConfig)
    vae_training: VAETrainingConfig = field(default_factory=VAETrainingConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        def _checked(dc_cls, data: dict) -> dict:
            valid = {f.name for f in fields(dc_cls)}
            unknown = set(data) - valid
            if unknown:
                raise ValueError(
                    f"Unknown keys for {dc_cls.__name__}: {sorted(unknown)}. "
                    f"Valid keys: {sorted(valid)}"
                )
            return data

        return cls(
            seed=d.get("seed", 42),
            paths=PathConfig(**_checked(PathConfig, d.get("paths", {}))),
            encoder=EncoderConfig(**_checked(EncoderConfig, d.get("encoder", {}))),
            decoder=DecoderConfig(**_checked(DecoderConfig, d.get("decoder", {}))),
            vae_arch=VAEArchConfig(**_checked(VAEArchConfig, d.get("vae_arch", {}))),
            vae_training=VAETrainingConfig(
                **_checked(VAETrainingConfig, d.get("vae_training", {}))
            ),
            quality_gate=QualityGateConfig(
                **_checked(QualityGateConfig, d.get("quality_gate", {}))
            ),
        )

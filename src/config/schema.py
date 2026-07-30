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
    prefix_inject: bool = True  # Route z into the K soft-prompt PREFIX tokens
    # (latent_proj(z) + prefix_pos_embed). False = ablation: the K prefix
    # positions stay (architecture/shape unchanged) but carry no z-dependent
    # info (prefix_pos_embed only) — isolates the prefix injection channel for
    # leave-one-out ablation against latent_pos_inject/deep_inject.
    latent_pos_inject: bool = True  # Add a K-pooled projection of z to EVERY
    # decoder token embedding (not just the K-token prefix). Keeps z reachable at
    # every step so the frozen decoder cannot ignore it.
    use_lora: bool = True  # Attach LoRA adapters to the decoder. A fully frozen
    # decoder has no trainable path to learn to read the injected latent and so
    # bypasses z (fluent but unconditioned output); LoRA gives it that path — the
    # primary cure for strong-decoder latent bypass. <1% trainable params.
    lora_r: int = 32  # LoRA rank.
    lora_alpha: int = 64  # LoRA scaling.
    lora_dropout: float = 0.05  # LoRA dropout.
    deep_inject: bool = True  # Per-layer KV injection (Optimus/LangVAE-style):
    # project the K latent tokens into K key/value memory slots in EVERY decoder
    # layer's attention (past_key_values), in addition to the input-layer prefix.
    # Raises the z→decoder bandwidth by ~n_layer×, the main lever for high-
    # fidelity reconstruction (BLEU). False = input-layer prefix + context only.
    kv_fanout_len: int = 0  # Full-sequence KV fan-out (LangVAE's W_m mechanism).
    # 0 = OFF (the K latent tokens map 1-to-1 to K KV slots/layer, current
    # behaviour). >0 = pool the latent to ONE vector and FAN it into this many
    # KV (key,value) slots PER decoder layer, so a single (K=1) vector gives the
    # frozen decoder ~kv_fanout_len read-points/layer instead of 1. This is the
    # bit-efficiency lever that lets LangVAE read a ~2-nat latent into BLEU 0.76.
    # Set to ~max_answer_len. Cost: kv_proj = latent_dim × (n_layer×2×n_kv_head×
    # head_dim×kv_fanout_len). At 50: ~118M on GPT-2, ~370M on Llama-3B, ~423M on
    # Mistral-7B (GQA keeps n_kv_head small) — all trainable, same order as LangVAE's
    # own W_m. Only active when deep_inject=True.
    torch_dtype: str = "float32"  # Load dtype for the decoder LM: "float32",
    # "float16", or "bfloat16". Use "bfloat16" for large decoders (Mistral/Llama)
    # on A100 — fp32 would need ~4× the memory. The trained injection heads stay
    # fp32; their outputs are cast to this dtype at injection time.
    load_in_4bit: bool = False  # 4-bit NF4 quantization (QLoRA) for the frozen
    # decoder backbone — needed to fit 3B/7B decoders. Requires bitsandbytes and
    # use_lora=True (adapters train in the compute dtype over the quantized base).
    device_map: Optional[str] = None  # HF device_map for the decoder (e.g.
    # "auto" to shard a large model across GPUs). None keeps it on the default
    # device (the training loop's .to(device)).
    fanout_mode: str = "auto"  # How the KV fan-out reaches the decoder:
    # "kv" = per-layer key/value memory (past_key_values), high bandwidth. On
    #        absolute-position decoders (GPT-2) the keys are injected raw. On
    #        ROTARY (RoPE) decoders (Qwen/Llama/Mistral) the injected KEYS are
    #        automatically rotated to their cache positions (using the model's own
    #        rotary module) so they align with the rotated queries — otherwise the
    #        latent is unreadable. This is the preferred, high-fidelity path.
    # "prefix" = the fan-out is projected into soft-prompt PREFIX embeddings
    #        prepended to the input (input-layer only, lower bandwidth). RoPE-safe
    #        fallback; worked but capped reconstruction near 0 on RoPE decoders.
    # "auto" = pick "prefix" if the decoder uses RoPE, else "kv" (safe default).
    #        RoPE combos set "kv" explicitly to use the RoPE-aware per-layer path.
    #        Only active when deep_inject=True and kv_fanout_len>0.


@dataclass(frozen=True)
class VAEArchConfig:
    """VAE architecture hyperparameters."""

    latent_dim: int = 128  # Per-token latent dimensionality.
    embed_dim: int = 768  # (Unused by the frozen backbones; kept for compat.)
    num_layers: int = 4  # (Unused — encoder is a frozen pretrained backbone.)
    decoder_num_layers: Optional[int] = None  # (Unused — decoder is a frozen
    # pretrained causal LM; depth is fixed by the backbone.)
    num_heads: int = 8  # Heads for the Perceiver cross-attention pool.
    dropout: float = 0.1  # Dropout for the Perceiver pool.
    max_answer_len: int = 50  # Max answer token length (encoder side).
    num_latent_tokens: int = 16  # K latent tokens (the sequence latent). K*D =
    # 16*128 = 2048 latent dims — capacity for near-lossless encoding of up to
    # ~50 tokens. K=4 (512 dims) was the information bottleneck behind BLEU ~20.
    latent_pos_inject: bool = True  # (Kept for compat; the decoder reads
    # ``decoder.latent_pos_inject``.)
    use_bow_head: bool = False  # Build a bag-of-words head that predicts the
    # answer's token set from z alone (Zhao et al. 2017). OFF — anti-collapse
    # reserve lever; not needed in the near-AE regime. Trained via
    # ``vae_training.bow_loss_weight`` when enabled.


@dataclass(frozen=True)
class VAETrainingConfig:
    """VAE training hyperparameters."""

    dataset: str = "squad_v2"  # VAE training corpus selector. "squad_v2" =
    # reconstruct SQuAD v2 answer texts (short, with NULLs). "entailment_bank" =
    # reconstruct EntailmentBank explanatory sentences (full declarative sentences,
    # no NULLs), matching the LangVAE paper (arXiv:2505.00004): all explanatory
    # ("cot") sentences, deduped, 99/1 split, one sentence per example.
    learning_rate: float = 1e-3  # AdamW peak LR. Only the small trained heads/
    # adapters see gradients, so a higher LR than a full-model train is fine.
    batch_size: int = 50  # Training batch size
    epochs: int = 50  # Maximum training epochs
    patience: int = 10  # Early stopping patience (val checks)
    warmup_steps: int = 500  # LR scheduler warmup
    weight_decay: float = 0.01  # AdamW weight decay
    grad_clip_max_norm: float = 5.0  # Gradient clipping threshold. 1.0 clipped a
    # ~38 pre-clip grad_norm ~38x → effective LR throttled, convergence crawled.
    grad_accum_steps: int = 1  # Gradient accumulation steps
    # --- KL regime: near-autoencoder (latent-diffusion recipe) ---
    # Goal is high-fidelity reconstruction (BLEU ~90): the latent must carry
    # hundreds of nats, so KL pressure is tiny — like Stable Diffusion's VAE.
    # The downstream diffusion model learns the prior; exported latents are
    # normalized (normalization_stats.pt), so a non-smooth N(0,I) fit is OK.
    # Prior-sampling quality (decode z ~ N(0,I) directly) is traded away.
    beta_start: float = 0.0  # KL weight at the start of warmup.
    beta_end: float = 0.01  # Max KL weight. Tiny on purpose (see regime note).
    # 0.05 left the posterior std ~0.84 → eval-time z=mu lost low-redundancy bits
    # → content-word substitutions capped EM ~35 / BLEU ~70. 0.01 gives more bits
    # for rare/content tokens. Watch val/f1_gap (collapse guard) + latent/std_mean.
    beta_warmup_steps: int = 5000  # Steps to ramp beta (monotonic schedule).
    beta_schedule: str = "monotonic"  # "monotonic" or "cyclical". Cyclical was an
    # anti-collapse measure; at beta 0.01 there is no collapse pressure to cycle.
    beta_cycles: int = 40  # (unused with monotonic)
    target_kl: Optional[float] = None  # KL hinge target (None = disabled). A
    # finite value actively pushes the latent back toward that few-nat budget —
    # the opposite of what high-fidelity reconstruction needs.
    beta_cycle_ratio: float = 0.5  # (unused with monotonic)
    free_bits: float = 0.0  # OFF — no collapse pressure at beta 0.01, nothing to
    # floor. (Per-dim KL allowance, Kingma 2016; reserve lever only.)
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
    word_dropout: float = 0.2  # Probability of replacing each teacher-forced
    # decoder INPUT token with a RANDOM token during training (Bowman et al.
    # 2016). Removes the teacher-forcing crutch so the decoder must read z to
    # predict tokens 2…N (anti-bypass). 0.5→0.2: with deep injection + near-AE KL
    # the decoder no longer bypasses z (free-running eval self-enforces z-usage),
    # so a light dose for exposure robustness only. 0.0 disables.
    bow_loss_weight: float = 0.0  # Weight on the bag-of-words auxiliary loss
    # (requires vae_arch.use_bow_head). Added directly to the total loss like a
    # second reconstruction term; uses the same per-sequence-sum reduction so
    # the weight is comparable to recon. 0.0 disables. ~0.3-1.0 is typical.
    zforce_weight: float = 0.3  # Weight on the z-forcing auxiliary pass (Goyal
    # et al. 2017). A SECOND teacher-forced decode of the SAME decoder run with
    # word_dropout=1.0 — every input token is [MASK], so the decoder must
    # reconstruct from z (+ position) ALONE. Trains the deployed decoder under
    # generation-like conditions (no gold prev-token bypass) while the primary
    # word_dropout=word_dropout pass preserves inter-token fluency. The direct
    # cure for the teacher-forced/generation gap (good train recon, EM/F1 ~0).
    # Added to total like a second recon term (same reduction), so the weight is
    # comparable to recon. 0.0 disables. Requires a mask token id (BERT [MASK]).
    consistency_weight: float = 0.0  # Weight on the self-distillation consistency
    # loss (Hinton et al. 2015, applied intra-model). Distills the fluent MAIN
    # (teacher-forced) pass into the z-only pass: KL(stopgrad(softmax(main_logits))
    # || softmax(zonly_logits)), reusing the SAME z-only forward as zforce (no extra
    # pass). Where zforce_weight matches the z-only pass to the HARD gold token, this
    # matches it to the teacher's FULL soft distribution — a far richer signal that
    # forces z to carry enough of the sentence to reproduce the teacher's fluency.
    # For a meaningful teacher, keep the main pass's word_dropout low (e.g. 0.3) so
    # it has real gold context. Same reduction as recon, so comparable to recon/
    # zforce weights. 0.0 disables. Requires a mask token id (BERT [MASK]).
    consistency_temp: float = 1.0  # Softmax temperature T for the consistency loss.
    # T=1.0 = raw distributions; T>1 softens both, exposing more "dark knowledge"
    # (relative weights of plausible tokens). The loss is scaled by T^2 (standard
    # distillation scaling) so its gradient magnitude stays comparable across T.


@dataclass(frozen=True)
class QualityGateConfig:
    """Thresholds for latent quality gate checks."""

    min_recon_accuracy: float = 0.85  # Minimum token reconstruction accuracy
    min_mean_kl: float = 1.0  # Minimum mean KL (trivially passed in the near-AE
    # regime — KL should sit in the hundreds of nats)
    min_active_dims: int = 64  # Minimum active latent dimensions (~3% of
    # K*D = 16*128 = 2048)
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

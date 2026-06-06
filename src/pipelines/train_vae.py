"""Training pipeline for the Sequence VAE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.config.schema import Config
from src.models.vae.vae import SequenceVAE
from src.models.vae.loss import compute_beta, compute_cyclical_beta
from src.training.ema import EMAManager
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.grad_utils import clip_gradients, accumulation_step
from src.training.checkpoint import save_checkpoint
from src.utils.logging import init_wandb, log_wandb, log_wandb_table, finish_wandb

logger = logging.getLogger(__name__)


def _validate(
    vae: SequenceVAE,
    val_loader,
    device: torch.device,
    beta: float = 1.0,
    free_bits: float = 0.0,
    target_kl: float | None = None,
    bow_weight: float = 0.0,
    tokenizer=None,
) -> dict[str, float]:
    """Run one pass over val_loader and return averaged metrics.

    When *tokenizer* is provided also computes reconstruction EM and F1:
    the VAE encodes each answer then decodes it; EM/F1 measure how faithfully
    the decoded text matches the original answer.
    """
    from src.evaluation.squad_metrics import compute_squad_metrics

    vae.eval()
    totals: dict[str, float] = {
        "total": 0.0, "recon": 0.0, "kl": 0.0, "bow": 0.0, "true_kl": 0.0,
    }
    n_batches = 0
    all_preds: list[str] = []
    all_refs: list[list[str]] = []
    # Latent-ablation probe: decode each example a SECOND time from a wrong
    # latent (μ rolled by one within the batch). If predictions barely change
    # vs the real-μ decode, the decoder is ignoring z (posterior collapse); a
    # large f1 gap means z genuinely drives the output. This is the only honest
    # collapse signal — KL can sit pinned at the free_bits floor regardless.
    all_preds_shuf: list[str] = []

    # Use the deterministic latent (μ) for evaluation and the model's own
    # autoregressive decoder so EM/F1 reflect real inference behaviour rather
    # than teacher-forced argmax at the ground-truth length.
    #
    # The data layer appends [SEP] (sep_token_id) as the end-of-answer marker
    # (see squad_dataset.py), so generation must stop on [SEP]. A BERT
    # tokenizer has no eos_token (eos_token_id is None), so reading
    # eos_token_id would disable early stopping entirely — generation would
    # emit all max_answer_len tokens and the trailing junk after [SEP] drives
    # EM/F1 to ~0. Prefer sep_token_id, fall back to eos_token_id.
    eos_token_id = None
    if tokenizer is not None:
        eos_token_id = getattr(tokenizer, "sep_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
    gen_max_len = vae.config.max_answer_len

    with torch.no_grad():
        for batch in val_loader:
            answer_ids = batch["answer_ids"].to(device)
            answer_mask = batch["answer_mask"].to(device)
            logits, _, mu, log_var, loss_dict = vae(
                answer_ids,
                answer_mask,
                beta=beta,
                free_bits=free_bits,
                target_kl=target_kl,
                bow_weight=bow_weight,
            )
            for k in ("total", "recon", "kl", "bow"):
                totals[k] += loss_dict[k].item()
            # Raw KL (no free-bits floor) to detect posterior collapse.
            # mu/log_var are (B, K, D); mean over batch then sum over (K, D).
            kl_raw = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            totals["true_kl"] += kl_raw.mean(dim=0).sum().item()
            n_batches += 1

            if tokenizer is not None and "all_answer_texts" in batch:
                # Autoregressively decode from μ. No ground-truth length leak.
                pred_ids = vae.decode_to_tokens(
                    mu, strategy="greedy", max_len=gen_max_len, eos_token_id=eos_token_id
                )  # (B, max_len)

                # Decode again from a shuffled (wrong) latent for the ablation
                # probe. Roll by one within the batch so every example gets a
                # mismatched μ; a 1-element batch can't be shuffled so reuse μ.
                mu_shuf = torch.roll(mu, shifts=1, dims=0) if mu.size(0) > 1 else mu
                pred_ids_shuf = vae.decode_to_tokens(
                    mu_shuf, strategy="greedy", max_len=gen_max_len,
                    eos_token_id=eos_token_id,
                )

                for i in range(pred_ids.size(0)):
                    pred_text = tokenizer.decode(
                        pred_ids[i].tolist(), skip_special_tokens=True
                    ).strip()
                    all_preds.append(pred_text)
                    all_refs.append(batch["all_answer_texts"][i])
                    all_preds_shuf.append(
                        tokenizer.decode(
                            pred_ids_shuf[i].tolist(), skip_special_tokens=True
                        ).strip()
                    )

    if n_batches == 0:
        return totals

    result = {k: v / n_batches for k, v in totals.items()}

    if tokenizer is not None and all_preds:
        squad = compute_squad_metrics(all_preds, all_refs)
        result["em"] = squad["em"]
        result["f1"] = squad["f1"]
        result["has_ans_em"] = squad["has_ans_em"]
        result["has_ans_f1"] = squad["has_ans_f1"]

        # Latent-ablation probe: f1 of the wrong-latent decode against the same
        # references, and the gap vs the real-latent f1. f1_gap ≈ 0 → decoder
        # ignores z (collapse); large positive gap → z drives the output. Use
        # f1_gap (not KL) as the trustworthy anti-collapse signal.
        if all_preds_shuf:
            squad_shuf = compute_squad_metrics(all_preds_shuf, all_refs)
            result["f1_shuffled"] = squad_shuf["f1"]
            result["f1_gap"] = squad["f1"] - squad_shuf["f1"]
            result["has_ans_f1_shuffled"] = squad_shuf["has_ans_f1"]
            result["has_ans_f1_gap"] = squad["has_ans_f1"] - squad_shuf["has_ans_f1"]

        # BLEU over ANSWERABLE examples only — the failing case we're chasing
        # (nulls have empty references and would just inflate the score). Multi-
        # reference, normalized identically to EM/F1. 0-100 scale.
        from src.evaluation.text_metrics import compute_bleu_multiref

        ans_preds, ans_refs = [], []
        for pred, golds in zip(all_preds, all_refs):
            if bool(golds) and any(g.strip() for g in golds):
                ans_preds.append(pred)
                ans_refs.append(golds)
        bleu = compute_bleu_multiref(ans_preds, ans_refs)
        result["has_ans_bleu"] = bleu.get("bleu", float("nan"))

        # Stash a few decoded samples so the caller can log them to W&B for
        # eyeballing what the decoder actually generates. Prefer answerable
        # examples (the failing case) but include a couple of nulls too.
        from src.evaluation.squad_metrics import token_f1

        rows: list[list] = []
        ans_rows, null_rows = [], []
        for pred, golds in zip(all_preds, all_refs):
            gold0 = golds[0] if golds else ""
            answerable = bool(golds) and any(g.strip() for g in golds)
            f1 = max((token_f1(pred, g) for g in golds), default=0.0) if answerable else float(pred.strip() == "")
            row = [gold0, pred, "yes" if answerable else "no", round(f1, 3)]
            (ans_rows if answerable else null_rows).append(row)
        rows = ans_rows[:20] + null_rows[:5]
        result["_samples"] = rows  # type: ignore[assignment]

    return result


def train_vae(
    config: Config,
    device: Optional[torch.device] = None,
    train_loader=None,
    val_loader=None,
) -> dict[str, float]:
    """Train the SequenceVAE and return final validation metrics.

    Parameters
    ----------
    config : Config
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    train_loader : DataLoader, optional
        If provided, skips building a dataloader from SQuAD (useful for tests).
    val_loader : DataLoader, optional
        If provided, skips building a dataloader from SQuAD (useful for tests).

    Returns
    -------
    dict
        Final val metrics: ``{total, recon, kl}``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.config.seed import seed_everything

    seed_everything(config.seed)

    # ------------------------------------------------------------------ data
    from src.data.tokenization import create_tokenizer

    if train_loader is None or val_loader is None:
        from src.data.loaders import create_vae_dataloaders

        tokenizer = create_tokenizer(config.encoder.model_name)
        # Subsample NULL examples in the VAE training set (and skip the
        # answerability-balanced sampler) so reconstruction gradient focuses on
        # real answer text. Export/classifier/diffusion still use the full set.
        # Dispatches on vae_training.dataset; null_train_fraction is ignored for
        # the EntailmentBank corpus (it has no NULLs).
        train_loader, val_loader = create_vae_dataloaders(
            config,
            tokenizer,
            null_train_fraction=config.vae_training.null_train_fraction,
        )
    else:
        tokenizer = create_tokenizer(config.encoder.model_name)

    # ------------------------------------------------------------------ model
    # Use the same tokenizer that the data loaders use so vocab_size includes
    # the [NULL_ANS] special token added by create_tokenizer.
    vocab_size = len(tokenizer)

    # Load real pretrained embeddings from the encoder model (e.g. BERT).
    # Random init forces the VAE to relearn the vocabulary subspace from
    # scratch and is a major cause of poor reconstruction.
    from src.utils.pretrained_embeddings import load_pretrained_token_embeddings

    pretrained_emb = load_pretrained_token_embeddings(
        model_name=config.encoder.model_name,
        target_vocab_size=vocab_size,
        target_embed_dim=config.vae_arch.embed_dim,
    ).to(device)

    vae = SequenceVAE(config.vae_arch, pretrained_embeddings=pretrained_emb).to(device)

    tc = config.vae_training

    # Token id used to corrupt teacher-forced decoder inputs for word dropout.
    # Prefer [MASK]; fall back to [UNK] (both exist in BERT) so the feature is
    # never silently disabled by a missing id.
    word_dropout_id = getattr(tokenizer, "mask_token_id", None)
    if word_dropout_id is None:
        word_dropout_id = getattr(tokenizer, "unk_token_id", None)
    if tc.word_dropout > 0.0 and word_dropout_id is None:
        logger.warning(
            "word_dropout=%.2f requested but tokenizer has no mask/unk token id; "
            "word dropout will be inactive.", tc.word_dropout,
        )

    # Separate weight decay groups: exclude biases, log_tau, and the
    # variational heads. Biases shouldn't be decayed (standard practice) and
    # log_tau needs to grow freely. Critically, ``mu_head``/``logvar_head``
    # are EXCLUDED: weight decay on them pulls their weights toward zero, which
    # drives μ→0 and log_var→0, i.e. q(z|x)→N(0,I). When the decoder isn't yet
    # using z, that decay is the dominant force and it actively collapses the
    # posterior (true_kl observed crashing to ~0 with all dims dead). Removing
    # decay here lets reconstruction + BoW gradients decide the posterior.
    # output_head.linear.weight stays in the decay group (tied to the embedding,
    # kept bounded).
    _no_decay = {"bias", "log_tau", "mu_head", "logvar_head"}
    param_groups = [
        {
            "params": [
                p
                for n, p in vae.named_parameters()
                if not any(nd in n for nd in _no_decay) and p.requires_grad
            ],
            "weight_decay": tc.weight_decay,
        },
        {
            "params": [
                p
                for n, p in vae.named_parameters()
                if any(nd in n for nd in _no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = create_optimizer(
        param_groups, lr=tc.learning_rate, weight_decay=tc.weight_decay
    )

    # total_steps should count optimizer updates, not raw batches, so that the
    # scheduler step() calls (inside accumulation_step) map correctly onto the
    # declared schedule length regardless of grad_accum_steps.
    steps_per_epoch = max(len(train_loader), 1)
    total_optimizer_steps = (tc.epochs * steps_per_epoch) // max(tc.grad_accum_steps, 1)
    scheduler = create_scheduler(optimizer, tc.warmup_steps, total_optimizer_steps)

    ema = EMAManager(vae, decay=tc.ema_decay, start_step=0)

    # ------------------------------------------------------------------ wandb
    init_wandb(config.to_dict(), project="latent-diffusion-text-vae")

    # Select the best checkpoint on val has_ans_f1 (real autoregressive
    # generation quality), NOT on val total loss. Total loss is dominated by the
    # teacher-forced reconstruction term, which keeps dropping as the decoder
    # memorises the corpus marginal — entirely independent of whether z is used.
    # Selecting on it happily saves a fully posterior-collapsed model (EM/F1 ~0).
    # has_ans_f1 reflects generation from z alone, so it tracks actual z-use.
    best_val_f1 = -float("inf")
    patience_counter = 0
    global_step = 0
    final_metrics: dict[str, float] = {}
    total_training_steps = tc.epochs * steps_per_epoch

    for epoch in range(tc.epochs):
        vae.train()

        # Accumulators for epoch-averaged train metrics
        epoch_totals: dict[str, float] = {
            "loss": 0.0,
            "recon": 0.0,
            "kl": 0.0,
            "bow": 0.0,
            "recon_zonly": 0.0,
            "true_kl": 0.0,
            "grad_norm": 0.0,
            "mu_mean": 0.0,
            "mu_std": 0.0,
            "std_mean": 0.0,
            "std_std": 0.0,
            "z_mean": 0.0,
            "z_std": 0.0,
            "active_dims": 0.0,
            "kl_per_dim_max": 0.0,
            "kl_per_dim_min": 0.0,
            "kl_per_dim_std": 0.0,
            "kl_per_dim_mean": 0.0,
            "dead_dims": 0.0,
        }
        epoch_steps = 0

        for batch in train_loader:
            global_step += 1
            epoch_steps += 1
            answer_ids = batch["answer_ids"].to(device)
            answer_mask = batch["answer_mask"].to(device)

            if tc.beta_schedule == "cyclical":
                beta = compute_cyclical_beta(
                    global_step,
                    total_steps=total_training_steps,
                    start=tc.beta_start,
                    end=tc.beta_end,
                    n_cycles=tc.beta_cycles,
                    ratio=tc.beta_cycle_ratio,
                )
            else:
                beta = compute_beta(
                    global_step,
                    start=tc.beta_start,
                    end=tc.beta_end,
                    warmup_steps=tc.beta_warmup_steps,
                )

            # Decoder noise augmentation: with probability noise_aug_prob,
            # perturb z by extra Gaussian noise inside vae.forward. Trains
            # the decoder to tolerate the slightly-imperfect latents that
            # diffusion sampling will produce at inference time.
            if tc.noise_aug_sigma > 0.0 and tc.noise_aug_prob > 0.0:
                aug_sigma = (
                    tc.noise_aug_sigma
                    if torch.rand(1).item() < tc.noise_aug_prob
                    else 0.0
                )
            else:
                aug_sigma = 0.0

            # Downweight NULL (unanswerable) examples in the reconstruction loss
            # so the decoder's gradient focuses on real answer text. Answerable
            # examples keep weight 1.0; nulls get tc.null_loss_weight. Nulls are
            # still encoded/decoded, so their latents remain available for export.
            recon_weights = None
            if "is_answerable" in batch and tc.null_loss_weight != 1.0:
                is_answerable = batch["is_answerable"].to(device)
                recon_weights = torch.ones(
                    answer_ids.size(0), device=device
                )
                recon_weights[~is_answerable] = tc.null_loss_weight

            logits, z, mu, log_var, loss_dict = vae(
                answer_ids,
                answer_mask,
                beta=beta,
                free_bits=tc.free_bits,
                target_kl=tc.target_kl,
                noise_aug_sigma=aug_sigma,
                recon_weights=recon_weights,
                word_dropout=tc.word_dropout,
                mask_token_id=word_dropout_id,
                bow_weight=tc.bow_loss_weight,
                zforce_weight=tc.zforce_weight,
            )
            loss = loss_dict["total"] / tc.grad_accum_steps
            loss.backward()

            grad_norm = 0.0
            if accumulation_step(global_step, tc.grad_accum_steps):
                grad_norm = clip_gradients(vae, tc.grad_clip_max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                ema.update(global_step)

            # Accumulate train metrics for epoch average
            with torch.no_grad():
                # mu, log_var, z are sequence-shaped: (B, K, D). Active dims
                # and KL stats are computed over the flattened (K, D) feature
                # axis so each "dim" is one (k, d) coordinate.
                std = torch.exp(0.5 * log_var)  # (B, K, D)
                if mu.shape[0] > 1:
                    mu_flat = mu.reshape(mu.size(0), -1)  # (B, K*D)
                    active_dims = int((mu_flat.var(dim=0) > 0.01).sum().item())
                else:
                    active_dims = 0
                kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, K, D)
                kl_per_dim_mean = kl_per_dim.mean(dim=0).reshape(-1)  # (K*D,)
                # True KL (no free bits) — exposes posterior collapse
                true_kl = kl_per_dim_mean.sum().item()

            step_metrics = {
                "loss": loss_dict["total"].item(),
                "recon": loss_dict["recon"].item(),
                "kl": loss_dict["kl"].item(),
                "bow": loss_dict["bow"].item(),
                "recon_zonly": loss_dict["recon_zonly"].item(),
                "true_kl": true_kl,
                "grad_norm": grad_norm,
                "mu_mean": mu.mean().item(),
                "mu_std": mu.std().item(),
                "std_mean": std.mean().item(),
                "std_std": std.std().item(),
                "z_mean": z.mean().item(),
                "z_std": z.std().item(),
                "active_dims": active_dims,
                "kl_per_dim_max": kl_per_dim_mean.max().item(),
                "kl_per_dim_min": kl_per_dim_mean.min().item(),
                "kl_per_dim_std": kl_per_dim_mean.std().item(),
                # Direct collapse signals: average information per latent
                # coordinate, and how many coordinates are effectively dead
                # (<0.01 nats). With healthy training kl_per_dim_mean stays
                # well above free_bits and dead_dims stays low.
                "kl_per_dim_mean": kl_per_dim_mean.mean().item(),
                "dead_dims": int((kl_per_dim_mean < 0.01).sum().item()),
            }

            for k, v in step_metrics.items():
                epoch_totals[k] += v

            # ---- wandb: per-step live curves ----
            log_wandb(
                {
                    "train/loss": step_metrics["loss"],
                    "train/recon": step_metrics["recon"],
                    "train/kl": step_metrics["kl"],
                    "train/bow": step_metrics["bow"],
                    "train/recon_zonly": step_metrics["recon_zonly"],
                    "train/true_kl": step_metrics["true_kl"],
                    "train/grad_norm": step_metrics["grad_norm"],
                    "train/beta": beta,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "latent/mu_mean": step_metrics["mu_mean"],
                    "latent/mu_std": step_metrics["mu_std"],
                    "latent/std_mean": step_metrics["std_mean"],
                    "latent/std_std": step_metrics["std_std"],
                    "latent/z_mean": step_metrics["z_mean"],
                    "latent/z_std": step_metrics["z_std"],
                    "latent/active_dims": step_metrics["active_dims"],
                    "latent/kl_per_dim_max": step_metrics["kl_per_dim_max"],
                    "latent/kl_per_dim_min": step_metrics["kl_per_dim_min"],
                    "latent/kl_per_dim_std": step_metrics["kl_per_dim_std"],
                    "latent/kl_per_dim_mean": step_metrics["kl_per_dim_mean"],
                    "latent/dead_dims": step_metrics["dead_dims"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            # ---- intra-epoch validation every val_every_n_steps ----
            if global_step % tc.val_every_n_steps == 0:
                # Validate with EMA weights for a smoother, more stable estimate.
                # Use tc.beta_end so the val loss composition is fixed across all
                # epochs (early-epoch beta≈0 would make val loss look artificially
                # good and cause premature early stopping).
                ema.apply()
                val_metrics = _validate(
                    vae,
                    val_loader,
                    device,
                    beta=tc.beta_end,
                    free_bits=tc.free_bits,
                    target_kl=tc.target_kl,
                    bow_weight=tc.bow_loss_weight,
                    tokenizer=tokenizer,
                )
                ema.restore()
                # Pull decoded samples out of the metrics dict and log them as a
                # W&B table so they don't leak into the scalar metric logging.
                samples = val_metrics.pop("_samples", None)
                if samples:
                    log_wandb_table(
                        "val/samples",
                        columns=["gold", "prediction", "answerable", "f1"],
                        rows=samples,
                        step=global_step,
                    )
                    # Also surface a couple in the console log for quick sanity.
                    for gold, pred, ans, f1 in samples[:5]:
                        logger.info("  sample [ans=%s f1=%.2f] gold=%r pred=%r",
                                    ans, f1, gold, pred)
                final_metrics = val_metrics
                logger.info(
                    "step=%d  val_loss=%.4f  recon=%.4f  kl=%.4f  em=%.3f  f1=%.3f  "
                    "f1_shuf=%.3f  f1_gap=%.3f",
                    global_step,
                    val_metrics["total"],
                    val_metrics["recon"],
                    val_metrics["kl"],
                    val_metrics.get("em", float("nan")),
                    val_metrics.get("f1", float("nan")),
                    val_metrics.get("f1_shuffled", float("nan")),
                    val_metrics.get("f1_gap", float("nan")),
                )
                log_wandb(
                    {
                        "val/loss": val_metrics["total"],
                        "val/recon": val_metrics["recon"],
                        "val/kl": val_metrics["kl"],
                        "val/bow": val_metrics["bow"],
                        "val/true_kl": val_metrics["true_kl"],
                        "val/em": val_metrics.get("em", float("nan")),
                        "val/f1": val_metrics.get("f1", float("nan")),
                        "val/has_ans_em": val_metrics.get("has_ans_em", float("nan")),
                        "val/has_ans_f1": val_metrics.get("has_ans_f1", float("nan")),
                        "val/has_ans_bleu": val_metrics.get("has_ans_bleu", float("nan")),
                        "val/f1_shuffled": val_metrics.get("f1_shuffled", float("nan")),
                        "val/f1_gap": val_metrics.get("f1_gap", float("nan")),
                        "val/has_ans_f1_gap": val_metrics.get("has_ans_f1_gap", float("nan")),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

                current_f1 = val_metrics.get("has_ans_f1", float("nan"))
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    patience_counter = 0
                    ckpt_path = Path(config.paths.checkpoint_dir) / "vae_best.pt"
                    ema.apply()
                    save_checkpoint(
                        path=ckpt_path,
                        model=vae,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ema=ema,
                        config=config,
                        step=global_step,
                        metrics=val_metrics,
                    )
                    ema.restore()
                    logger.info("Saved VAE checkpoint (EMA weights): %s", ckpt_path)
                else:
                    # Only accrue patience once beta has finished ramping.
                    # During warmup the training objective is still changing
                    # (KL pressure rising), so a recon plateau here is expected
                    # and must NOT end the run — otherwise early stopping can
                    # fire before beta ever reaches beta_end, checkpointing a
                    # model optimised for a weak-KL objective. `beta` is the
                    # value used for this step's training pass (in scope from
                    # the batch loop above).
                    if beta >= tc.beta_end - 1e-9:
                        patience_counter += 1
                    else:
                        logger.info(
                            "step=%d  val plateau ignored (beta=%.4f < beta_end=%.4f, "
                            "still warming up)", global_step, beta, tc.beta_end,
                        )

                # Early stopping is only armed after beta warmup completes, so
                # the warmup phase always runs to completion regardless of when
                # val loss plateaus.
                if beta >= tc.beta_end - 1e-9 and patience_counter > tc.patience:
                    logger.info("Early stopping at step %d", global_step)
                    finish_wandb()
                    return final_metrics

                vae.train()

        # ---- end-of-epoch summary ----
        n = max(epoch_steps, 1)
        logger.info(
            "epoch=%d  train_loss=%.4f  recon=%.4f  kl=%.4f",
            epoch,
            epoch_totals["loss"] / n,
            epoch_totals["recon"] / n,
            epoch_totals["kl"] / n,
        )

    # Ensure a final validation pass ran even if the last step wasn't a val step
    if not final_metrics:
        ema.apply()
        final_metrics = _validate(
            vae, val_loader, device,
            beta=tc.beta_end, free_bits=tc.free_bits,
            target_kl=tc.target_kl, bow_weight=tc.bow_loss_weight,
            tokenizer=tokenizer,
        )
        ema.restore()
        final_metrics.pop("_samples", None)

    finish_wandb()
    return final_metrics


if __name__ == "__main__":
    from src.config.loader import create_config_from_cli

    logging.basicConfig(level=logging.INFO)
    cfg = create_config_from_cli()
    metrics = train_vae(cfg)
    print("Final metrics:", metrics)

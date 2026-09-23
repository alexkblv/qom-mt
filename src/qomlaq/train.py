"""Fine-tuning one direction of one split.

Training and evaluation are deliberately separate stages. Training only produces a
checkpoint plus its model card; scoring happens in :mod:`qomlaq.evaluate` against
whatever was saved. The released artifact and the evaluated artifact are the same object
by construction, and a re-score never needs a retrain.

Early stopping is on by default, and the best checkpoint by dev loss is the one kept.
"""

from __future__ import annotations

from pathlib import Path

from .config import TRAINING, ModelProfile, TrainingConfig
from .evaluate import ModelCard
from .experiments import RunSpec
from .splits import Split


class TrainingError(RuntimeError):
    pass


def build_datasets(split: Split, run: RunSpec, tokenizer, cfg: TrainingConfig):
    """Tokenize train and dev for one direction.

    Uses ``text_target=``, which is the supported way to tokenize labels for NLLB: it
    applies the target language tag without mutating tokenizer state between calls.
    """
    from datasets import Dataset

    def encode(batch):
        tokenizer.src_lang = run.src_lang
        tokenizer.tgt_lang = run.tgt_lang
        return tokenizer(
            batch["src"],
            text_target=batch["tgt"],
            max_length=cfg.max_length,
            truncation=True,
        )

    out = {}
    for name in ("train", "dev"):
        frame = split.partition(name)
        dataset = Dataset.from_dict(
            {
                "src": frame[run.src_col].astype(str).tolist(),
                "tgt": frame[run.tgt_col].astype(str).tolist(),
            }
        )
        out[name] = dataset.map(encode, batched=True, remove_columns=["src", "tgt"])
    return out["train"], out["dev"]


def truncation_report(split: Split, run: RunSpec, tokenizer, cfg: TrainingConfig) -> dict:
    """How much of the training data does not fit the model window.

    Bible verses are long, and a target cut mid-verse teaches the model to stop early.
    """
    counts = {}
    for name in ("train", "dev", "test"):
        frame = split.partition(name)
        tokenizer.src_lang = run.src_lang
        src = tokenizer(frame[run.src_col].astype(str).tolist(), add_special_tokens=True)["input_ids"]
        tokenizer.src_lang = run.tgt_lang
        tgt = tokenizer(frame[run.tgt_col].astype(str).tolist(), add_special_tokens=True)["input_ids"]
        counts[name] = {
            "rows": len(frame),
            "src_over_limit": int(sum(1 for i in src if len(i) > cfg.max_length)),
            "tgt_over_limit": int(sum(1 for i in tgt if len(i) > cfg.max_length)),
            "src_p95_tokens": int(sorted(len(i) for i in src)[int(0.95 * len(src))]) if src else 0,
            "tgt_p95_tokens": int(sorted(len(i) for i in tgt)[int(0.95 * len(tgt))]) if tgt else 0,
        }
    return counts


def train_run(
    run: RunSpec,
    split: Split,
    output_dir: str | Path,
    *,
    profile: ModelProfile,
    model_key: str = "",
    cfg: TrainingConfig = TRAINING,
    init_from: str | None = None,
    init_label: str | None = None,
    resume_from_checkpoint: bool | str = False,
) -> ModelCard:
    """Fine-tune one direction and save the checkpoint with its model card.

    Training starts from ``profile.hf_id`` unless ``init_from`` names another checkpoint
    (a Hugging Face id or a local model directory) to continue from. ``init_label`` is
    what the model card records for it, e.g. ``baseline/base__stratified__es2qom`` in
    place of a local path.
    """
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        NllbTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    if split.split_id != run.split_id:
        raise TrainingError(
            f"run {run.run_id} expects split {run.split_id} but was given {split.split_id}"
        )

    output_dir = Path(output_dir)
    work_dir = output_dir / "checkpoints"
    model_dir = output_dir / "model"

    source = init_from or profile.hf_id
    tokenizer = NllbTokenizer.from_pretrained(source)
    model = AutoModelForSeq2SeqLM.from_pretrained(source)
    if profile.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_ds, dev_ds = build_datasets(split, run, tokenizer, cfg)
    truncation = truncation_report(split, run, tokenizer, cfg)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    args = Seq2SeqTrainingArguments(
        output_dir=str(work_dir),
        per_device_train_batch_size=profile.per_device_batch_size,
        per_device_eval_batch_size=profile.per_device_batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        fp16=profile.fp16 and torch.cuda.is_available(),
        optim=cfg.optim,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=cfg.seed,
        # Generation during training would cost a full decode of dev every epoch while
        # selection is on loss; the test decode happens once, in the evaluate stage.
        predict_with_generate=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    best = min(history, key=lambda h: h["eval_loss"]) if history else {}

    # load_best_model_at_end has already restored the best weights in memory.
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    card = ModelCard.for_run(
        run,
        split,
        base_model=profile.hf_id,
        model=model_key,
        init_from=init_label or init_from,
        training={
            **{k: v for k, v in vars(cfg).items()},
            "per_device_batch_size": profile.per_device_batch_size,
            "gradient_accumulation_steps": profile.gradient_accumulation_steps,
            "effective_batch_size": profile.effective_batch_size,
            "fp16": profile.fp16,
            "truncation": truncation,
        },
        selection={
            "criterion": "eval_loss",
            "best_epoch": best.get("epoch"),
            "best_eval_loss": best.get("eval_loss"),
            "epochs_run": len(history),
            "early_stopping_patience": cfg.early_stopping_patience,
            "eval_loss_by_epoch": [
                {"epoch": h.get("epoch"), "eval_loss": h.get("eval_loss")} for h in history
            ],
        },
    )
    card.save(model_dir)
    return card

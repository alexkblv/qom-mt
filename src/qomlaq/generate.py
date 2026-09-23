"""Translation.

NLLB is conditioned through the tokenizer's ``src_lang`` and a forced BOS token for the
target language -- never through text prefixes.

Order is preserved end to end: hypotheses come back in the same sequence as the inputs,
because the scorer pairs them positionally with the references.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import GENERATION, GenerationConfig


class GenerationError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    device: str

    @property
    def dtype(self) -> str:
        return str(getattr(self.model, "dtype", "unknown"))


def load_model(path: str | Path, *, device: str | None = None) -> LoadedModel:
    """Load a checkpoint (a local directory or a Hugging Face id) for inference."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

    resolved = str(path)
    local = Path(resolved).exists()
    tokenizer = NllbTokenizer.from_pretrained(resolved, local_files_only=local)
    model = AutoModelForSeq2SeqLM.from_pretrained(resolved, local_files_only=local)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


def forced_bos_id(tokenizer, tgt_lang: str) -> int:
    """Resolve the target-language token id, failing loudly if the tag is unknown."""
    token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    unk = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk:
        raise GenerationError(
            f"tokenizer has no token for language tag {tgt_lang!r}; generation would be "
            "conditioned on <unk> and silently produce the wrong language"
        )
    return token_id


def count_truncated(
    tokenizer, texts: Sequence[str], src_lang: str, max_length: int
) -> int:
    """How many inputs exceed the model's window.

    Reported with every evaluation: a truncated source is scored against a full-length
    reference, which depresses the score for reasons unrelated to translation quality.
    """
    tokenizer.src_lang = src_lang
    encoded = tokenizer(list(texts), add_special_tokens=True)["input_ids"]
    return int(sum(1 for ids in encoded if len(ids) > max_length))


def translate(
    texts: Sequence[str],
    loaded: LoadedModel,
    *,
    src_lang: str,
    tgt_lang: str,
    config: GenerationConfig = GENERATION,
    progress: bool = True,
) -> list[str]:
    """Translate ``texts``, preserving order."""
    import torch

    tokenizer, model = loaded.tokenizer, loaded.model
    tokenizer.src_lang = src_lang
    bos = forced_bos_id(tokenizer, tgt_lang)

    outputs: list[str] = []
    total = len(texts)
    for start in range(0, total, config.batch_size):
        batch = [str(t) for t in texts[start : start + config.batch_size]]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_source_length,
        ).to(loaded.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=bos,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        if progress and (start // config.batch_size) % 20 == 0:
            print(f"  translated {min(start + config.batch_size, total)}/{total}", flush=True)

    if len(outputs) != total:
        raise GenerationError(
            f"produced {len(outputs)} hypotheses for {total} inputs; order or count was lost"
        )
    return outputs

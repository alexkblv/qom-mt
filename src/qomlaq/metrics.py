"""Scoring.

No metric name is ever written by hand. A name is derived from the sacrebleu parameters
by :func:`chrf_display_name`, so ``word_order=2, beta=1`` renders as ``chrF1++`` and
cannot be mistaken for the standard metric, and the standard metric can only be produced
by the parameters that define it. Every score also carries sacrebleu's own signature
string, which a test cross-checks against the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

MetricKind = Literal["CHRF", "BLEU"]


def chrf_display_name(char_order: int = 6, word_order: int = 0, beta: int = 2) -> str:
    """sacrebleu's own naming convention: ``chrF{beta}`` plus one ``+`` per word order.

    >>> chrf_display_name(word_order=2, beta=2)
    'chrF2++'
    >>> chrf_display_name(word_order=2, beta=1)
    'chrF1++'
    >>> chrf_display_name(word_order=0, beta=2)
    'chrF2'
    """
    name = f"chrF{beta}" + "+" * word_order
    return name if char_order == 6 else f"{name}[c{char_order}]"


def bleu_display_name(tokenize: str | None = None, **_: Any) -> str:
    return "BLEU" if tokenize in (None, "13a") else f"BLEU[{tokenize}]"


@dataclass(frozen=True)
class MetricSpec:
    kind: MetricKind
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        if self.kind == "CHRF":
            return chrf_display_name(**self.kwargs)
        return bleu_display_name(**self.kwargs)

    def build(self):
        return CHRF(**self.kwargs) if self.kind == "CHRF" else BLEU(**self.kwargs)


#: The reported metric. chrF++ as defined by Popovic (2017) and used by the AmericasNLP
#: shared tasks: character 6-grams plus word unigrams and bigrams, beta=2.
PRIMARY = MetricSpec("CHRF", {"char_order": 6, "word_order": 2, "beta": 2})

#: Reported alongside for comparability with prior work.
SECONDARY: tuple[MetricSpec, ...] = (
    MetricSpec("CHRF", {"char_order": 6, "word_order": 0, "beta": 2}),  # plain chrF
    MetricSpec("BLEU", {}),                                             # BLEU, 13a
)

DEFAULT_METRICS: tuple[MetricSpec, ...] = (PRIMARY, *SECONDARY)


@dataclass(frozen=True)
class ScoreSet:
    scores: dict[str, float]
    signatures: dict[str, str]
    n_pairs: int

    @property
    def primary(self) -> float:
        return self.scores[PRIMARY.display_name]

    @property
    def primary_name(self) -> str:
        return PRIMARY.display_name

    def as_dict(self) -> dict:
        return {
            "scores": dict(self.scores),
            "signatures": dict(self.signatures),
            "n_pairs": self.n_pairs,
            "primary_metric": PRIMARY.display_name,
            "sacrebleu_version": sacrebleu.__version__,
        }


class MetricError(RuntimeError):
    pass


def score(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    metrics: Sequence[MetricSpec] = DEFAULT_METRICS,
) -> ScoreSet:
    """Corpus-level scoring of detokenized text.

    Hypotheses and references must be positionally aligned and equal in length; a
    mismatch raises rather than silently scoring a shifted pairing.
    """
    if len(hypotheses) != len(references):
        raise MetricError(
            f"hypotheses ({len(hypotheses)}) and references ({len(references)}) differ in "
            "length; they must be positionally aligned"
        )
    if not hypotheses:
        raise MetricError("nothing to score")

    hyps = [str(h) for h in hypotheses]
    refs = [[str(r) for r in references]]

    scores: dict[str, float] = {}
    signatures: dict[str, str] = {}
    for spec in metrics:
        metric = spec.build()
        result = metric.corpus_score(hyps, refs)
        name = spec.display_name
        scores[name] = round(float(result.score), 4)
        signatures[name] = metric.get_signature().format(short=False)
    return ScoreSet(scores=scores, signatures=signatures, n_pairs=len(hyps))


def score_by_group(
    hypotheses: Sequence[str],
    references: Sequence[str],
    groups: Sequence[str],
    *,
    metric: MetricSpec = PRIMARY,
) -> dict[str, float]:
    """Per-source (or per-any-label) breakdown of the primary metric.

    Base+Bible test sets are ~90% Bible, so a single corpus score mostly reports
    Bible-register performance; this is what makes that visible.
    """
    buckets: dict[str, tuple[list[str], list[str]]] = {}
    for hyp, ref, key in zip(hypotheses, references, groups):
        h, r = buckets.setdefault(str(key), ([], []))
        h.append(str(hyp))
        r.append(str(ref))
    out: dict[str, float] = {}
    for key, (hyps, refs) in sorted(buckets.items()):
        built = metric.build()
        out[key] = round(float(built.corpus_score(hyps, [refs]).score), 4)
    return out

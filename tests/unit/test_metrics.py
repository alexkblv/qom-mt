"""Metric names and scores.

A metric's name is derived from its parameters, so a chrF variant can never be labeled
as a different one.
"""

from __future__ import annotations

import re

import pytest
import sacrebleu

from qomlaq.metrics import (
    DEFAULT_METRICS,
    PRIMARY,
    MetricError,
    MetricSpec,
    chrf_display_name,
    score,
    score_by_group,
)


@pytest.mark.parametrize(
    "word_order,beta,expected",
    [(2, 2, "chrF2++"), (2, 1, "chrF1++"), (0, 2, "chrF2"), (0, 1, "chrF1"), (1, 2, "chrF2+")],
)
def test_display_name_is_derived_from_parameters(word_order, beta, expected):
    assert chrf_display_name(word_order=word_order, beta=beta) == expected


def test_primary_metric_is_standard_chrf_pp():
    """chrF++ is char 6-grams + word 1..2-grams with beta=2 (Popovic 2017, AmericasNLP)."""
    assert PRIMARY.kwargs == {"char_order": 6, "word_order": 2, "beta": 2}
    assert PRIMARY.display_name == "chrF2++"


def test_other_chrf_parameterizations_do_not_render_as_chrf_pp():
    """Other chrF variants must be visibly different metrics."""
    chrf1_pp = MetricSpec("CHRF", {"word_order": 2, "beta": 1})
    chrf2 = MetricSpec("CHRF", {"word_order": 0, "beta": 2})  # sacrebleu's default
    assert chrf1_pp.display_name == "chrF1++" != PRIMARY.display_name
    assert chrf2.display_name == "chrF2" != PRIMARY.display_name


def test_sacrebleu_default_is_not_chrf_pp():
    """Guards the specific mistake: corpus_chrf() with no arguments is plain chrF."""
    default = sacrebleu.metrics.CHRF()
    assert default.word_order == 0
    hyp, ref = ["el gato negro"], [["el gato blanco"]]
    assert sacrebleu.corpus_chrf(hyp, ref).score != PRIMARY.build().corpus_score(hyp, ref).score


def test_signature_matches_spec_parameters():
    """Label, parameters and library must agree; the signature is the cross-check."""
    result = score(["el gato negro corre"], ["el gato blanco corre"])
    signature = result.signatures[PRIMARY.display_name]
    assert re.search(r"nw:(\d+)", signature).group(1) == str(PRIMARY.kwargs["word_order"])
    assert re.search(r"nc:(\d+)", signature).group(1) == str(PRIMARY.kwargs["char_order"])
    assert f"version:{sacrebleu.__version__}" in signature


def test_every_default_metric_reports_a_signature():
    result = score(["a b c"], ["a b d"])
    for spec in DEFAULT_METRICS:
        assert spec.display_name in result.scores
        assert spec.display_name in result.signatures


def test_score_rejects_misaligned_inputs():
    """An off-by-one between hypotheses and references must not score silently."""
    with pytest.raises(MetricError):
        score(["a", "b"], ["a"])


def test_score_rejects_empty_input():
    with pytest.raises(MetricError):
        score([], [])


def test_perfect_and_disjoint_scores_bracket_the_range():
    perfect = score(["exactamente igual"], ["exactamente igual"])
    assert perfect.primary == pytest.approx(100.0, abs=1e-6)
    poor = score(["xxxxx"], ["completamente distinto"])
    assert poor.primary < 20.0


def test_score_by_group_partitions_the_corpus():
    hyps = ["a b c", "d e f", "g h i"]
    refs = ["a b c", "d e x", "g h i"]
    per_source = score_by_group(hyps, refs, ["A", "B", "A"])
    assert set(per_source) == {"A", "B"}
    assert per_source["A"] == pytest.approx(100.0, abs=1e-6)
    assert per_source["B"] < 100.0

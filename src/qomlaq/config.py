"""Declarative registry: languages, sources, corpora, models and the fixed recipes.

Everything the pipeline knows about the shape of the data lives here, as data rather
than as code branching inside a loader. Adding a source is a registry entry; it cannot
be done by writing a bespoke reader in a notebook.

What gets *run* -- which corpus, splits, model and directions -- is not declared here but
in an experiment file under ``experiments/`` (see :mod:`qomlaq.experiments`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SEED = 42

LANG_QOM = "grn_Latn"  # Guarani proxy tag; NLLB-200 has no code for Qom (ISO tob)
LANG_ES = "spa_Latn"


@dataclass(frozen=True)
class Direction:
    """One translation direction: which column is the source, and the NLLB tags."""

    id: str
    src_col: str
    tgt_col: str
    src_lang: str
    tgt_lang: str
    label: str


DIRECTIONS: dict[str, Direction] = {
    "es2qom": Direction("es2qom", "es", "qom", LANG_ES, LANG_QOM, "ES→QOM"),
    "qom2es": Direction("qom2es", "qom", "es", LANG_QOM, LANG_ES, "QOM→ES"),
}

Terminal = Literal["error", "document"]
Kind = Literal["xlsx", "csv"]


@dataclass(frozen=True)
class GroupRule:
    """How to derive a discourse-unit key for a source.

    ``levels`` are id columns tried in order; the first non-null wins. ``terminal``
    says what happens when every level is null: ``"document"`` puts the row in a
    single whole-document group, ``"error"`` raises. There is deliberately no
    per-line terminal: one-row groups would let neighboring lines of the same unit land
    on both sides of a split.
    """

    levels: tuple[str, ...]
    terminal: Terminal = "error"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    filename: str
    kind: Kind
    col_qom: str
    col_es: str
    group: GroupRule
    sheet: str | None = None
    #: Structural id columns to carry through from the source, if present.
    id_columns: tuple[str, ...] = ()
    #: Id columns built by joining existing ones with "_", as (name, (columns...)) pairs,
    #: e.g. a book-and-chapter key.
    derived_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    #: Rows are verses (``libro``/``capitulo``/``versiculo``). Verse-level checks and the
    #: verse-level grouping control apply to them.
    verses: bool = False
    #: Sheets holding ``nombre_*_qom`` / ``nombre_*_es`` title pairs.
    title_sheets: tuple[str, ...] = ()
    #: Corpus configurations this source belongs to.
    configs: tuple[str, ...] = ()
    #: If set, every row is forced into this partition and never appears in train.
    forced_partition: str | None = None
    #: Template for a human-readable reference, formatted with the row's fields.
    ref_template: str = "{source_slug}_{row}"

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "_").replace("'", "")


_XLSX_IDS = ("id_linea", "id_fragmento", "id_seccion", "id_capitulo")

SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="Arte verbal qom",
        filename="Arte verbal qom.xlsx",
        kind="xlsx",
        sheet="Lineas",
        col_qom="linea_qom",
        col_es="linea_es",
        id_columns=_XLSX_IDS,
        title_sheets=("Capítulos", "Fragmentos"),
        # 451 rows (chapters intro, c5, c6) carry no fragment annotation; chapter is
        # the finest structure that actually exists for them.
        group=GroupRule(levels=("id_fragmento", "id_seccion", "id_capitulo")),
        configs=("base", "base_bible"),
        ref_template="arteverbal_{row}",
    ),
    SourceSpec(
        name="Educación Sanitaria Intercultural",
        filename="Educación Sanitaria Intercultural.xlsx",
        kind="xlsx",
        sheet="Lineas",
        col_qom="linea_qom",
        col_es="linea_es",
        id_columns=_XLSX_IDS,
        title_sheets=("Capítulos", "Fragmentos"),
        group=GroupRule(levels=("id_fragmento", "id_seccion", "id_capitulo")),
        configs=("base", "base_bible"),
        ref_template="salud_{row}",
    ),
    SourceSpec(
        name="Materiales del Taller de Lengua y Cultura Toba",
        filename="Materiales del Taller de Lengua y Cultura Toba.xlsx",
        kind="xlsx",
        sheet="Lineas",
        col_qom="linea_qom",
        col_es="linea_es",
        id_columns=("id_linea", "id_fragmento", "id_seccion"),
        title_sheets=("Secciones", "Fragmentos"),
        group=GroupRule(levels=("id_fragmento", "id_seccion")),
        configs=("base", "base_bible"),
        ref_template="taller_{row}",
    ),
    SourceSpec(
        name="Las Aventuras de Copaic",
        filename="Las Aventuras de Copaic.xlsx",
        kind="xlsx",
        sheet="Sheet1",
        col_qom="linea_qom",
        col_es="linea_es",
        id_columns=("id_linea",),
        # A 16-line children's story with no internal structure. Held out whole as a
        # declared out-of-domain probe, and reported separately rather than folded
        # into the headline test metric.
        group=GroupRule(levels=(), terminal="document"),
        configs=("base", "base_bible"),
        forced_partition="test",
        ref_template="copaic_{row}",
    ),
    SourceSpec(
        name="La Declaración Universal de los Derechos Humanos",
        filename="La Declaración Universal de los Derechos Humanos.xlsx",
        kind="xlsx",
        sheet="Sheet1",
        col_qom="linea_qom",
        col_es="linea_es",
        # 'parte' is genuine article-level structure.
        id_columns=("id_linea", "parte"),
        group=GroupRule(levels=("parte",), terminal="document"),
        configs=("base", "base_bible"),
        ref_template="udhr_{row}",
    ),
    SourceSpec(
        name="El Principito",
        filename="El Principito.csv",
        kind="csv",
        col_qom="qom",
        col_es="espanol",
        id_columns=("capitulo",),
        group=GroupRule(levels=("capitulo",)),
        configs=("base", "base_bible"),
        ref_template="principito_cap{capitulo}_{row}",
    ),
    SourceSpec(
        name="La Biblia",
        filename="La Biblia.csv",
        kind="csv",
        col_qom="LNLE13",
        col_es="DHHS94",
        id_columns=("libro", "capitulo", "versiculo"),
        derived_ids=(("libro_capitulo", ("libro", "capitulo")),),
        verses=True,
        # Chapter-level grouping, so neighboring verses never straddle a split.
        group=GroupRule(levels=("libro_capitulo",)),
        configs=("base_bible", "bible_only"),
        ref_template="{libro} {capitulo}:{versiculo}",
    ),
)

SOURCES_BY_NAME: dict[str, SourceSpec] = {s.name: s for s in SOURCES}

#: Named corpus versions: the sources a corpus is built from. An experiment names one,
#: so adding a source means adding a version, and older experiments keep building the
#: exact corpus they were run on.
CORPORA: dict[str, tuple[str, ...]] = {
    "text-v1": tuple(s.name for s in SOURCES),
}


def corpus_sources(corpus: str) -> tuple[SourceSpec, ...]:
    if corpus not in CORPORA:
        raise KeyError(f"unknown corpus {corpus!r}; expected one of {sorted(CORPORA)}")
    return tuple(SOURCES_BY_NAME[name] for name in CORPORA[corpus])


def configs_of(sources: tuple[SourceSpec, ...]) -> tuple[str, ...]:
    """Corpus configurations the given sources belong to, in declaration order."""
    return tuple(dict.fromkeys(c for s in sources for c in s.configs))


#: Every configuration any source belongs to.
CORPUS_CONFIGS: tuple[str, ...] = configs_of(SOURCES)


def config_has_verses(config: str) -> bool:
    """Whether a configuration includes verse-structured sources (the Bible)."""
    return any(s.verses and config in s.configs for s in SOURCES)


@dataclass(frozen=True)
class QCConfig:
    """Filters applied to every source identically."""

    min_tokens: int = 2
    len_ratio_min: float = 0.15
    len_ratio_max: float = 6.0
    max_chars: int = 800
    dedup_scope: Literal["within_source", "global"] = "within_source"


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.80
    dev: float = 0.10
    test: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "dev": self.dev, "test": self.test}


SPLIT_STRATEGIES: tuple[str, ...] = ("random", "stratified")
BIBLE_GROUPINGS: tuple[str, ...] = ("chapter", "verse")
SPLIT_PURPOSES: tuple[str, ...] = ("headline", "control")

#: One ratio for every configuration. QomL-Base is small enough that 80/10/10 leaves a
#: thin test set, but a per-config ratio would make splits harder to compare.
DEFAULT_SPLITS = SplitConfig()


def split_dir_name(config: str, strategy: str, bible_grouping: str = "chapter") -> str:
    """Split id, also its directory name. Bible grouping is explicit wherever it applies."""
    if not config_has_verses(config):
        return f"{config}/{strategy}"
    return f"{config}/{strategy}__{bible_grouping}"


@dataclass(frozen=True)
class ModelProfile:
    """A base checkpoint plus the memory settings it needs to train.

    Batch size and precision depend on the model's size and the GPU, so they live here
    and not in the training recipe. The effective batch should stay the same across
    profiles unless an experiment is meant to change it.
    """

    hf_id: str
    per_device_batch_size: int
    gradient_accumulation_steps: int
    fp16: bool = True
    gradient_checkpointing: bool = True

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


MODELS: dict[str, ModelProfile] = {
    "nllb-600m": ModelProfile(
        hf_id="facebook/nllb-200-distilled-600M",
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
    ),
}


@dataclass(frozen=True)
class GenerationConfig:
    """Frozen decoding configuration, shared by every evaluation and by the translator,
    so all scores and all demo output come from the same decoding."""

    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    max_new_tokens: int = 256
    max_source_length: int = 256
    batch_size: int = 8

    def as_dict(self) -> dict[str, int]:
        return {
            "num_beams": self.num_beams,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "max_new_tokens": self.max_new_tokens,
            "max_source_length": self.max_source_length,
            "batch_size": self.batch_size,
        }


GENERATION = GenerationConfig()


@dataclass(frozen=True)
class TrainingConfig:
    """The training recipe. An experiment can override any field in its ``[training]``
    table; everything else comes from here."""

    learning_rate: float = 5e-4
    num_train_epochs: int = 10
    early_stopping_patience: int = 2
    max_length: int = 256
    optim: str = "adafactor"
    seed: int = SEED


TRAINING = TrainingConfig()

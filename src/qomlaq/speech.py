"""Turn the Ciudad Universitaria transcriptions (.docx) into the project's *formato
computacional* table, plus one review column:

    index | location | speaker | audio | qom | español | revisión

One row is one source segment and its Spanish translation, in source order, grouped by
speaker and then by audio file. ``revisión`` carries the review notes for that row, in
Spanish for the linguists. Rows with nothing to review have an empty ``revisión`` cell.

The source documents lay the material out as alternating paragraphs:

    <audio header>          e.g. "<Name> audio 1. 20:20" / "<Name Surname>, audio 5"
    <qom paragraph>
    <spanish paragraph>
    <qom paragraph>
    <spanish paragraph>
    ...

Nothing here corrects the Qom. Text is copied character for character (see STRIP_EDGES
and NORMALIZE_NBSP for the only two whitespace settings). Anything ambiguous is reported
in ``revisión`` for a linguist instead of being resolved silently.

Apostrophes: the glottal stop in these documents is U+2019, and they also use U+2018 and
U+00B4 in the same role. Those are not reported, since the project treats them as
orthographic variation. Decoding failures (U+FFFD, mojibake) are.

The documents hold transcriptions from named speakers and are not published, so they stay
in :func:`qomlaq.paths.speech_dir`, which git ignores. Each speaker's name is read from
the file name, and review notes about a speaker (``speaker_notes.json`` next to the
documents, ``{speaker: note}``) are read from there too.

Run:  python -m qomlaq.speech [--speech-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from . import paths

SOURCE_SUBDIR = "ciudad-universitaria"
DOCUMENT_RE = re.compile(r"^Transcripci[óo]n y traducci[óo]n audios (.+) Ciudad Universitaria\.docx$")
SPEAKER_NOTES = "speaker_notes.json"

LOCATION = "ciudad universitaria"          # as spelled in Viviana's sheet

# Strip leading/trailing whitespace from each cell. This removes docx justification
# artefacts (some source lines carry 100+ leading spaces). Interior characters are never
# touched.
STRIP_EDGES = True

# One document holds thousands of NO-BREAK SPACE (U+00A0) characters from a Google Docs
# export. They are kept by default: replacing them would substitute a lookalike character,
# which the project rules forbid.
NORMALIZE_NBSP = False

OUT_CORPUS = "corpus_ciudad_universitaria.csv"
# The per-issue listing, written next to the corpus table.
OUT_FLAGS = "revision_flags.csv"

COLUMNS = ["index", "location", "speaker", "audio", "qom", "español", "revisión"]

# Separator when a single row collects more than one review note.
NOTE_SEP = " | "

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# --------------------------------------------------------------------------
# Source documents
# --------------------------------------------------------------------------


def find_documents(source_dir: Path) -> list[tuple[Path, str]]:
    """(path, speaker) for each transcription document, the speaker taken from its name."""
    found = []
    for path in source_dir.iterdir():
        m = DOCUMENT_RE.match(unicodedata.normalize("NFC", path.name))
        if m:
            found.append((path, m.group(1)))
    if not found:
        sys.exit("No transcription documents in %s" % source_dir)
    return sorted(found, key=lambda item: unicodedata.normalize("NFC", item[0].name))


def read_speaker_notes(source_dir: Path) -> dict[str, str]:
    path = source_dir / SPEAKER_NOTES
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_paragraphs(path):
    """Return the document's paragraphs as plain text, in order."""
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(W + "body")
    out = []
    for p in body.findall(W + "p"):
        buf = []
        for node in p.iter():
            if node.tag == W + "t":
                buf.append(node.text or "")
            elif node.tag == W + "tab":
                buf.append("\t")
            elif node.tag == W + "br":
                buf.append("\n")
        out.append("".join(buf))
    return out


# --------------------------------------------------------------------------
# Structural line recognition
# --------------------------------------------------------------------------

SPACE = r"[\s ]*"

# "H-05_Name_01-02_040626_112853_Tr2.wav"
WAV_RE = re.compile(r"H-\d\d_[A-Za-z]+_\d\d-\d\d_\d+_\d+_Tr\d(?:\.wav)?\s*$")
NUMERIC_ONLY_RE = re.compile(r"^\d+$")


def _name_patterns(speakers: list[str]) -> list[str]:
    """A speaker's first name, optionally followed by the rest of the name."""
    patterns = []
    for speaker in speakers:
        first, *rest = speaker.split()
        pattern = re.escape(first)
        if rest:
            pattern += "(?:%s%s)?" % (SPACE, SPACE.join(map(re.escape, rest)))
        patterns.append(pattern)
    return patterns


def audio_header_re(speakers: list[str]) -> re.Pattern:
    """"<Name> audio 1. 20:20" / "<Name Surname>, audio 5" / "<Name>\\xa0<Surname>,audio\\xa010"."""
    return re.compile(r"^\s*(?:%s)%s,?%saudio%s(\d+)"
                      % ("|".join(_name_patterns(speakers)), SPACE, SPACE, SPACE), re.I)


def front_matter_re(speakers: list[str]) -> re.Pattern:
    firsts = "|".join(re.escape(s.split()[0]) for s in speakers)
    return re.compile(r"^(Grabaciones\b|Ciudad Universitaria$|Transcripci[óo]n\s+(?:%s))" % firsts,
                      re.I)


def clean(text):
    """Apply the two configured whitespace settings. Nothing else is altered."""
    if NORMALIZE_NBSP:
        text = text.replace(" ", " ")
    if STRIP_EDGES:
        text = re.sub(r"^[ \t\r\n ]+|[ \t\r\n ]+$", "", text)
    return text


# --------------------------------------------------------------------------
# Language identification (used only to detect anomalies, never to edit text)
# --------------------------------------------------------------------------

ES_WORDS = {
    "que", "de", "la", "el", "los", "las", "en", "un", "una", "y", "no", "para",
    "porque", "cuando", "entonces", "con", "se", "es", "está", "están", "muy",
    "más", "pero", "como", "sí", "su", "sus", "me", "nos", "lo", "al", "del",
    "por", "hay", "ya", "todo", "todos", "este", "esta", "eso", "ese", "son",
    "fue", "nuestro", "nuestra", "había", "tienen", "tiene", "donde", "cada",
    "sólo", "solo", "les", "le", "si", "a", "o", "yo", "mi", "era", "ahí",
}
QOM_WORDS = {
    "nache", "da", "na", "nam", "qom", "huo’o", "cha’aye", "so", "ca", "cam",
    "ne’ena", "se’eso", "qataq", "taq’en", "ýaqto’", "ishet", "qalaxaye", "dam",
    "ñi", "añi", "aso", "som", "qaq", "igo’", "ego’", "aja’", "qomi’", "de’eda",
    "nagui", "na’a’q", "yi", "asa’aso", "eda", "ýataqta", "sa", "ten", "mashe",
    "enac", "nataq’en", "qaica", "huotaique", "ýaýaten",
}
TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÿýỹñ’'\-]+")


def classify(text):
    """Return ('qom'|'es'|None, margin). None means 'too close to call'."""
    toks = TOKEN_RE.findall(text.lower())
    if not toks:
        return None, 0
    es = sum(1 for w in toks if w in ES_WORDS)
    qm = sum(1 for w in toks if w in QOM_WORDS)
    qm += 2 * text.count("ý")          # ý occurs only in the Qom tier
    if es == qm:
        return None, 0
    return ("es" if es > qm else "qom"), abs(es - qm)


def agrees(label, expected):
    """A 'too close to call' label is compatible with either expectation."""
    return label is None or label == expected


def other(tier):
    return "es" if tier == "qom" else "qom"


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def split_into_blocks(paragraphs, audio_re, front_re):
    """Group paragraphs into audio blocks. Returns (blocks, structural_notes)."""
    blocks, notes = [], []
    current, wav = None, None

    for idx, raw in enumerate(paragraphs):
        probe = raw.replace(" ", " ").strip()
        if not probe:
            continue
        if WAV_RE.search(probe):
            wav = probe
            continue
        m = audio_re.match(probe)
        if m:
            current = {
                "audio": "audio %d" % int(m.group(1)),
                "header": probe,
                "wav": wav,
                "items": [],
            }
            blocks.append(current)
            continue
        if current is None:
            if front_re.match(probe):
                continue
            notes.append((idx, probe, "texto suelto antes del primer encabezado de audio"))
            continue
        if NUMERIC_ONLY_RE.match(probe):
            notes.append((idx, probe, "párrafo con un número suelto"))
            continue
        current["items"].append((idx, raw))

    return blocks, notes


def pair_block(block, speaker, flags, next_index):
    """
    Pair a block's paragraphs into (qom, español) rows.

    Pairing follows the source's alternation. The language classifier is used only to
    spot where that alternation breaks:

      * a *sustained* disagreement (this paragraph AND the next one both contradict the
        expected tier) means a paragraph is missing or two tiers were merged into one
        line -> emit the dangling segment on its own, note it, and resynchronise.
      * an *isolated* disagreement keeps the positional pairing but notes the row, since
        heavy Spanish code-switching inside a Qom segment looks the same to the
        classifier as a genuine problem.
    """
    rows = []
    items = block["items"]
    audio = block["audio"]
    k, expect, held = 0, "qom", None

    def flag(row_index, issue):
        flags.append(
            {"row_index": row_index, "speaker": speaker, "audio": audio, "issue": issue}
        )

    while k < len(items):
        para_idx, raw = items[k]
        label, margin = classify(raw)

        if not agrees(label, expect) and margin >= 2:
            nxt = items[k + 1] if k + 1 < len(items) else None
            sustained = False
            if nxt is not None:
                nlabel, nmargin = classify(nxt[1])
                sustained = (not agrees(nlabel, other(expect))) and nmargin >= 2

            if sustained:
                if expect == "es" and held is not None:
                    rows.append((clean(held[1]), ""))
                    flag(
                        next_index + len(rows) - 1,
                        "Segmento qom sin traducción al español (párrafo fuente %d). La "
                        "alternancia qom/español se desplaza en este punto: la línea en "
                        "español contigua parece tener anexado el comienzo del siguiente "
                        "segmento qom. La celda «español» quedó vacía." % held[0],
                    )
                    held, expect = None, "qom"
                    continue                      # re-read the same paragraph as Qom
                rows.append(("", clean(raw)))
                flag(
                    next_index + len(rows) - 1,
                    "Segmento en español sin contraparte qom (párrafo fuente %d). La "
                    "alternancia qom/español se desplaza en este punto: la línea anterior "
                    "parece fusionar una traducción al español con el comienzo del siguiente "
                    "segmento qom. La celda «qom» quedó vacía." % para_idx,
                )
                k += 1
                expect = "qom"
                continue

        # Normal positional pairing.
        if expect == "qom":
            held = (para_idx, raw)
            expect = "es"
            k += 1
        else:
            qom_text = clean(held[1]) if held else ""
            es_text = clean(raw)
            rows.append((qom_text, es_text))
            row_index = next_index + len(rows) - 1

            qlabel, qmargin = classify(held[1]) if held else (None, 0)
            if qlabel == "es" and qmargin >= 2:
                ratio = difflib.SequenceMatcher(None, qom_text, es_text).ratio()
                if ratio >= 0.75:
                    flag(
                        row_index,
                        "Posible segmento duplicado: la celda «qom» (párrafo fuente %d) se lee "
                        "como español y es %.0f%% idéntica a la celda «español». O bien el "
                        "hablante pasó casi por completo al español, o la línea en español se "
                        "pegó dos veces y se perdió la línea qom."
                        % (held[0], ratio * 100),
                    )
                else:
                    flag(
                        row_index,
                        "La celda «qom» (párrafo fuente %d) no se lee como qom: se clasifica "
                        "como español. Puede ser alternancia de código intensa o una línea "
                        "fuera de lugar." % held[0],
                    )
            held, expect = None, "qom"
            k += 1

    if held is not None:
        rows.append((clean(held[1]), ""))
        flag(
            next_index + len(rows) - 1,
            "Segmento qom sin traducción al español: el párrafo fuente %d es el último de %s "
            "y no tiene línea de traducción a continuación." % (held[0], audio),
        )

    return rows


# --------------------------------------------------------------------------
# Character-level checks
# --------------------------------------------------------------------------

# Apostrophe / quotation-mark variants (U+2018, U+00B4, U+0027) are not reported: the
# project treats them as orthographic variation, not defects. Only decoding failures are
# reported here.
ODDITIES = {
    "�": "CARÁCTER DE REEMPLAZO (U+FFFD), indica un fallo de decodificación",
    "œ": "LIGADURA œ (U+0153), posible mojibake",
}


def extract(speech_dir: Path) -> None:
    source_dir = speech_dir / SOURCE_SUBDIR
    out_corpus = speech_dir / OUT_CORPUS
    out_flags = speech_dir / OUT_FLAGS

    documents = find_documents(source_dir)
    speakers = [speaker for _, speaker in documents]
    speaker_notes = read_speaker_notes(source_dir)
    audio_re, front_re = audio_header_re(speakers), front_matter_re(speakers)

    all_rows, flags, summary_blocks = [], [], []
    running = 1

    for path, speaker in documents:
        paragraphs = read_paragraphs(path)
        blocks, notes = split_into_blocks(paragraphs, audio_re, front_re)

        if speaker in speaker_notes:
            flags.append({
                "row_index": running,
                "speaker": speaker,
                "audio": "",
                "issue": speaker_notes[speaker],
            })

        for para_idx, text, kind in notes:
            flags.append({
                "row_index": running,
                "speaker": speaker,
                "audio": "",
                "issue": "Se encontró un %s en el párrafo fuente %d: «%s». No se incluyó como "
                         "segmento; confirmar que no es texto del corpus." % (kind, para_idx, text),
            })

        seen_audio = {}
        for block in blocks:
            rows = pair_block(block, speaker, flags, running)

            # Duplicate "audio N" labels within one speaker (one document's numbering
            # restarts partway through).
            if block["audio"] in seen_audio:
                flags.append({
                    "row_index": running,
                    "speaker": speaker,
                    "audio": block["audio"],
                    "issue": "Etiqueta de audio duplicada: «%s» ya se había usado para la "
                             "grabación «%s»; este bloque proviene de «%s» (encabezado «%s»). "
                             "Las dos grabaciones quedan bajo la misma etiqueta, así que los "
                             "conteos por audio las suman juntas. Hace falta decidir la "
                             "numeración." % (
                                 block["audio"], seen_audio[block["audio"]],
                                 block["wav"], block["header"]),
                })
            else:
                seen_audio[block["audio"]] = block["wav"]

            for qom_text, es_text in rows:
                all_rows.append({
                    "index": running,
                    "location": LOCATION,
                    "speaker": speaker,
                    "audio": block["audio"],
                    "qom": qom_text,
                    "español": es_text,
                })
                running += 1

            summary_blocks.append({
                "speaker": speaker, "audio": block["audio"],
                "wav": block["wav"] or "-", "rows": len(rows),
            })

    df = pd.DataFrame(all_rows, columns=COLUMNS[:-1])

    # ---- character-level checks ------------------------------------------
    for oddity, description in ODDITIES.items():
        hits = df[df["qom"].str.contains(oddity, regex=False)
                  | df["español"].str.contains(oddity, regex=False)]
        for _, r in hits.iterrows():
            flags.append({
                "row_index": r["index"], "speaker": r["speaker"], "audio": r["audio"],
                "issue": "Anomalía de codificación: %s. Se conserva tal cual." % description,
            })

    # NBSP is pervasive rather than exceptional - one aggregate note per speaker.
    for speaker in df["speaker"].unique():
        sub = df[df["speaker"] == speaker]
        nbsp = sub[sub["qom"].str.contains(" ", regex=False)
                   | sub["español"].str.contains(" ", regex=False)]
        if len(nbsp):
            flags.append({
                "row_index": int(nbsp.iloc[0]["index"]), "speaker": speaker,
                "audio": nbsp.iloc[0]["audio"],
                "issue": "ESPACIO DURO (U+00A0) presente en %d de las %d filas de este "
                         "hablante (la primera es esta). Se conserva tal cual; se señala "
                         "porque es un artefacto de exportación de Google Docs y no "
                         "ortografía. Poner NORMALIZE_NBSP=True en el script para "
                         "convertirlos en espacios normales." % (len(nbsp), len(sub)),
            })

    # Empty cells that no earlier rule already caught.
    already = {f["row_index"] for f in flags}
    for _, r in df.iterrows():
        if (not r["qom"].strip() or not r["español"].strip()) and r["index"] not in already:
            flags.append({
                "row_index": r["index"], "speaker": r["speaker"], "audio": r["audio"],
                "issue": "Celda vacía en esta fila: falta uno de los dos lados del par.",
            })

    # Bracketed / parenthesised material.
    for _, r in df.iterrows():
        for col in ("qom", "español"):
            for frag in re.findall(r"\[[^\]]*\]|\([^)]*\)", r[col]):
                flags.append({
                    "row_index": r["index"], "speaker": r["speaker"], "audio": r["audio"],
                    "issue": "Material entre corchetes o paréntesis «%s» en la celda «%s»: no "
                             "está claro si es texto del corpus o una anotación. Se dejó tal "
                             "cual." % (frag, col),
                })

    # Exact duplicate qom segments.
    non_empty = df[df["qom"].str.strip() != ""]
    dupes = non_empty[non_empty.duplicated(subset=["qom"], keep=False)]
    for _, grp in dupes.groupby("qom"):
        idxs = list(grp["index"])
        for i in idxs[1:]:
            r = df[df["index"] == i].iloc[0]
            flags.append({
                "row_index": i, "speaker": r["speaker"], "audio": r["audio"],
                "issue": "Segmento qom duplicado: el mismo texto aparece también en la(s) "
                         "fila(s) %s." % ", ".join(str(x) for x in idxs if x != i),
            })

    # ---- merge the review notes into the corpus as `revisión` ------------
    notes_by_row = defaultdict(list)
    for f in flags:
        if f["row_index"] != "":
            notes_by_row[int(f["row_index"])].append(f["issue"])

    df["revisión"] = df["index"].map(
        lambda i: NOTE_SEP.join(notes_by_row.get(int(i), []))
    )
    df = df[COLUMNS]

    flags_df = pd.DataFrame(flags, columns=["row_index", "speaker", "audio", "issue"])
    flags_df = flags_df.sort_values(
        by="row_index", key=lambda s: pd.to_numeric(s, errors="coerce").fillna(-1)
    ).reset_index(drop=True)

    df.to_csv(out_corpus, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    flags_df.to_csv(out_flags, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    report(df, flags_df, summary_blocks)
    roundtrip_check(df, out_corpus)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report(df, flags_df, summary_blocks):
    print("=" * 78)
    print("CIUDAD UNIVERSITARIA - parallel corpus summary")
    print("=" * 78)
    print("Total segments (rows): %d" % len(df))
    print()
    print("Rows per speaker")
    print("-" * 78)
    for speaker, n in df.groupby("speaker", sort=False).size().items():
        print("  %-16s %5d" % (speaker, n))
    print()
    print("Rows per audio file (in source order)")
    print("-" * 78)
    print("  %-16s %-10s %6s   %s" % ("speaker", "audio", "rows", "source recording"))
    for b in summary_blocks:
        print("  %-16s %-10s %6d   %s" % (b["speaker"], b["audio"], b["rows"], b["wav"]))
    print()
    empty_q = int((df["qom"].str.strip() == "").sum())
    empty_e = int((df["español"].str.strip() == "").sum())
    print("Rows with an empty qom cell:     %d" % empty_q)
    print("Rows with an empty español cell: %d" % empty_e)
    print()
    annotated = int((df["revisión"].str.strip() != "").sum())
    print("Review notes merged into the `revisión` column")
    print("-" * 78)
    print("  Notes written:            %d" % len(flags_df))
    print("  Rows carrying a note:     %d of %d" % (annotated, len(df)))
    print("  Rows with no note:        %d" % (len(df) - annotated))
    print()
    kinds = Counter(_kind(i) for i in flags_df["issue"])
    for kind, n in kinds.most_common():
        print("  %3d  %s" % (n, kind))
    print()


_KINDS = [
    ("Etiqueta de audio duplicada", "Etiqueta de audio duplicada"),
    ("Posible segmento duplicado", "Posible segmento duplicado"),
    ("Segmento qom sin traducción", "Segmento qom sin traducción al español"),
    ("Segmento en español sin contraparte", "Segmento en español sin contraparte qom"),
    ("no se lee como qom", "Celda «qom» que no se lee como qom"),
    ("ESPACIO DURO", "Espacio duro U+00A0 (nota agregada por hablante)"),
    ("Confirmar la grafía", "Grafía del nombre del hablante"),
    ("número suelto", "Párrafo con un número suelto"),
    ("Segmento qom duplicado", "Segmento qom duplicado"),
    ("Material entre corchetes", "Material entre corchetes o paréntesis"),
    ("Celda vacía", "Celda vacía"),
    ("Anomalía de codificación", "Anomalía de codificación"),
]


def _kind(issue):
    for needle, label in _KINDS:
        if needle in issue:
            return label
    return "otro"


def roundtrip_check(df, out_corpus: Path):
    """Re-read the CSV from disk and prove the special characters survived."""
    print("=" * 78)
    print("UTF-8 round-trip check")
    print("=" * 78)
    back = pd.read_csv(out_corpus, encoding="utf-8", keep_default_na=False, dtype=str)

    # Source text only - this is what must survive character for character.
    written = "".join(df["qom"]) + "".join(df["español"])
    read = "".join(back["qom"]) + "".join(back["español"])

    watch = ["’", "ý", "ñ", "ỹ", "í", "ó", "é",
             "á", "ú", "ü", "¿", "¡", "‘", "´",
             " ", "Ý", "Ñ"]
    print("  %-46s %8s %8s  %s" % ("character (in qom + español)", "written", "re-read", "ok"))
    ok = True
    for ch in watch:
        a, b = written.count(ch), read.count(ch)
        if a == 0 and b == 0:
            continue
        good = a == b
        ok &= good
        print("  U+%04X %-39s %8d %8d  %s"
              % (ord(ch), unicodedata.name(ch, "?"), a, b, "OK" if good else "MISMATCH"))

    same_shape = len(back) == len(df)
    identical = read == written
    rev_ok = "".join(back["revisión"]) == "".join(df["revisión"])
    cols_ok = list(back.columns) == COLUMNS
    print()
    print("  Column order preserved:      %s (%s)"
          % ("OK" if cols_ok else "MISMATCH", ", ".join(back.columns)))
    print("  Row count preserved:         %s (%d)" % ("OK" if same_shape else "MISMATCH", len(back)))
    print("  qom + español identical:     %s" % ("OK" if identical else "MISMATCH"))
    print("  revisión column identical:   %s" % ("OK" if rev_ok else "MISMATCH"))
    print("  Raw file decodes as UTF-8:   %s"
          % ("OK" if _decodes_utf8(out_corpus) else "MISMATCH"))
    print("  No U+FFFD after round trip:  %s" % ("OK" if "�" not in read else "MISMATCH"))
    print()
    print("  RESULT: %s"
          % ("PASS" if (ok and same_shape and identical and rev_ok and cols_ok) else "FAIL"))


def _decodes_utf8(path):
    try:
        path.read_bytes().decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qomlaq.speech", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speech-dir", default=None,
                        help="speech corpus sources (default: qomlaq.paths.speech_dir())")
    args = parser.parse_args(argv)
    extract(paths.speech_dir(args.speech_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

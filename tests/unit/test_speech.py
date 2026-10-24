"""Speech transcriptions to the formato computacional tables, on made-up documents."""

from __future__ import annotations

import json
import zipfile
from xml.sax.saxutils import escape

import pandas as pd

from qomlaq import speech

QOM = "nache qataq huo'o da ỹataqta"
ES = "entonces fue muy lindo para todos"
QOM_2 = "qalaxaye ishet da mashe huo'o"
ES_2 = "pero ya se puede porque hay"
QOM_3 = "qaq nataq'en ỹaỹaten"
PART_1 = "H-01_Pedro_01-02_010126_100000_Tr2.wav"
PART_2 = "H-01_Pedro_01-04_010126_100000_Tr2"


def write_docx(path, paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{escape(p)}</w:t></w:r></w:p>" for p in paragraphs)
    xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f"<w:body>{body}</w:body></w:document>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)


def document_name(speaker):
    return f"Transcripción y traducción audios {speaker} Ciudad Universitaria.docx"


def test_speakers_come_from_the_file_names(tmp_path):
    for speaker in ("Pedro Prueba", "Juana Ejemplo"):
        write_docx(tmp_path / document_name(speaker), [])
    (tmp_path / "notas.docx").write_bytes(b"")
    found = speech.find_documents(tmp_path)
    assert [speaker for _, speaker in found] == ["Juana Ejemplo", "Pedro Prueba"]


def test_a_part_starts_at_an_audio_file_name_on_its_own_line():
    assert speech.audio_name(PART_1) == PART_1
    assert speech.audio_name("M-03_Ana_01-02_010126_100000_Tr1") == "M-03_Ana_01-02_010126_100000_Tr1.wav"
    assert speech.audio_name(f"escuchamos {PART_1} ayer") is None


def test_the_classifier_reads_either_apostrophe_and_either_nasal_y():
    assert speech.classify(QOM)[0] == "qom"
    assert speech.classify(QOM.replace("'", "’").replace("ỹ", "ý"))[0] == "qom"
    assert speech.classify(ES)[0] == "es"


def test_extract_writes_line_pairs_and_one_row_per_audio_part(tmp_path):
    source = tmp_path / speech.SOURCE_SUBDIR
    source.mkdir()
    write_docx(source / document_name("Pedro Prueba"),
               ["Grabaciones Pedro Prueba", "Transcripción y traducción", PART_1, QOM, ES, QOM_2, ES_2,
                PART_2, QOM_3])
    (source / speech.SPEAKER_NOTES).write_text(json.dumps({"Pedro Prueba": "nota"}))

    speech.extract(tmp_path)

    lines = pd.read_csv(tmp_path / speech.OUT_CORPUS, keep_default_na=False, dtype=str)
    assert list(lines.columns) == speech.COLUMNS
    assert list(lines["audio"]) == [PART_1, PART_1, PART_2 + ".wav"]
    assert list(lines["español"]) == [ES, ES_2, ""]
    assert lines["revisión"][0] == "nota"
    assert "sin traducción" in lines["revisión"][2]

    parts = pd.read_csv(tmp_path / speech.OUT_PER_AUDIO, keep_default_na=False, dtype=str)
    assert list(parts.columns) == speech.PER_AUDIO_COLUMNS
    assert list(parts["renglones"]) == ["2", "1"]
    assert parts["qom"][0] == f"{QOM} {QOM_2}" and parts["español"][0] == f"{ES} {ES_2}"
    assert parts["revisión"][0] == "fila 1: nota"
    assert parts["revisión"][1].startswith("fila 3: Segmento qom sin traducción")

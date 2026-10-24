"""Speech transcriptions to the formato computacional table, on made-up documents."""

from __future__ import annotations

import json
import zipfile
from xml.sax.saxutils import escape

import pandas as pd

from qomlaq import speech

QOM = "nache qataq huo’o da ýataqta"
ES = "entonces fue muy lindo para todos"


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


def test_audio_headers_take_the_name_with_or_without_surname_comma_and_nbsp():
    header = speech.audio_header_re(["Juana Ejemplo", "Pedro Prueba"])
    for line in ("Juana audio 1. 20:20", "Juana Ejemplo, audio 5", "Pedro Prueba,audio 10"):
        assert header.match(line.replace(" ", " "))
    assert not header.match("Escuchamos el audio 2")


def test_extract_pairs_the_tiers_and_flags_a_missing_translation(tmp_path):
    source = tmp_path / speech.SOURCE_SUBDIR
    source.mkdir()
    write_docx(source / document_name("Juana Ejemplo"),
               ["Juana audio 1", QOM, ES, QOM, ES, "Juana audio 2", QOM])
    (source / speech.SPEAKER_NOTES).write_text(json.dumps({"Juana Ejemplo": "nota"}))

    speech.extract(tmp_path)

    table = pd.read_csv(tmp_path / speech.OUT_CORPUS, keep_default_na=False, dtype=str)
    assert list(table.columns) == speech.COLUMNS
    assert list(table["audio"]) == ["audio 1", "audio 1", "audio 2"]
    assert list(table["qom"]) == [QOM, QOM, QOM]
    assert list(table["español"]) == [ES, ES, ""]
    assert table["revisión"][0] == "nota"
    assert "sin traducción" in table["revisión"][2]

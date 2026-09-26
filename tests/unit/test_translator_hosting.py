"""The translator's Cloud Run deployment: what gets uploaded, installed and kept open.

Cloud Run bills an instance while any request to it is open, and the image is built
from the files cloudrun.sh archives. These checks read the sources; the test
environment doesn't install gradio.
"""

from __future__ import annotations

import re
from pathlib import Path

TRANSLATOR = Path(__file__).resolve().parents[2] / "translator"


def _read(name: str) -> str:
    return (TRANSLATOR / name).read_text(encoding="utf-8")


def test_app_keeps_no_connection_open_per_tab():
    # A gr.State or an unload event makes Gradio hold a heartbeat connection open
    # for as long as a tab stays open, which keeps the instance running and billed.
    source = _read("app.py")
    assert not re.search(r"\bgr\.State\(|\.unload\(", source)


def test_image_installs_the_pinned_requirements():
    dockerfile = _read("Dockerfile")
    assert "pip install -r translator/requirements.txt" in dockerfile
    assert not re.search(r"==\s*\d", dockerfile), "pin versions in requirements.txt"


def test_deploy_uploads_everything_the_image_copies():
    block = re.search(r"^FILES=\((.*?)\)", _read("cloudrun.sh"), re.S | re.M)
    uploaded = set(block.group(1).split())
    for line in re.findall(r"^COPY\s+(.+)$", _read("Dockerfile"), re.M):
        *sources, _destination = line.split()
        for source in sources:
            assert source in uploaded or source.split("/")[0] in uploaded, source

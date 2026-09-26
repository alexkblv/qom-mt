# The translator's image for Cloud Run. cloudrun.sh builds it from an archive of
# the committed files it needs, with this Dockerfile moved to the archive's root.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /srv

# The pinned dependencies first, so a code change doesn't reinstall torch.
COPY translator/requirements.txt translator/requirements.txt
RUN pip install -r translator/requirements.txt

COPY pyproject.toml README.md ./
COPY src src
RUN pip install .

COPY translator/app.py translator/app.py

RUN useradd --create-home translator
USER translator
WORKDIR /srv/translator

# Cloud Run sends requests to port 8080. QOM_PRELOAD=0 opens the page as soon
# as the server starts; the model loads on the first translation.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1 \
    QOM_PRELOAD=0

CMD ["python", "app.py"]

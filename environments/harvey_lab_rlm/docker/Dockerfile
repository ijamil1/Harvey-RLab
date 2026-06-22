FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        fontconfig \
        fonts-dejavu-core \
        fonts-liberation \
        libreoffice-calc \
        libreoffice-writer \
        pandoc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
        diff-match-patch==20241021 \
        docxtpl==0.20.2 \
        lxml==6.0.2 \
        openpyxl==3.1.5 \
        pandas==3.0.3 \
        python-docx==1.2.0 \
        xlcalculator==0.5.0

COPY docker/parse_doc.py /usr/local/bin/parse-doc
RUN chmod 0755 /usr/local/bin/parse-doc \
    && mkdir -p /workspace/output /workspace/skills /workspace/.lab

WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]

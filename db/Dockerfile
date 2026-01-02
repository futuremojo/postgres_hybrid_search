FROM postgres:17

ARG PG_TEXTSEARCH_VERSION=0.2.0

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        postgresql-17-pgvector \
        unzip; \
    arch="$(uname -m)"; \
    case "$arch" in \
        x86_64) pkg_arch="amd64" ;; \
        aarch64|arm64) pkg_arch="arm64" ;; \
        *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    url="https://github.com/timescale/pg_textsearch/releases/download/v${PG_TEXTSEARCH_VERSION}/pg-textsearch-v${PG_TEXTSEARCH_VERSION}-pg17-${pkg_arch}.zip"; \
    curl -L -o /tmp/pg_textsearch.zip "$url"; \
    unzip -q /tmp/pg_textsearch.zip -d /tmp/pg_textsearch; \
    dpkg -i /tmp/pg_textsearch/*.deb; \
    rm -rf /tmp/pg_textsearch /tmp/pg_textsearch.zip; \
    rm -rf /var/lib/apt/lists/*

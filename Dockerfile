FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml /app/pyproject.toml

RUN uv venv /app/.venv \
    && uv pip compile /app/pyproject.toml -o /tmp/requirements.txt \
    && uv pip install --python /app/.venv/bin/python -r /tmp/requirements.txt

ENV PATH="/app/.venv/bin:$PATH"

COPY app.py /app/app.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

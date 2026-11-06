FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
COPY model model
COPY app.py .
RUN useradd --system --uid 10001 app
USER app
ENV PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["python", "app.py"]

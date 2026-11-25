# Hugging Face Docker Space for the demo. Assembled by demo/deploy.py; build context = the
# assembled folder (src/, pyproject.toml, demo/, models/).
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only torch first, so the package's torch requirement does not pull CUDA wheels
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml README.md /app/
COPY --chown=user src /app/src
RUN pip install --no-cache-dir "/app[demo]"

COPY --chown=user demo /app/demo
COPY --chown=user models /app/models

USER user
ENV DEMO_BUNDLE=/app/demo/bundle DEMO_MODELS=/app/models DEMO_THREADS=2 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "demo.server.app:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]

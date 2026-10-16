# Registry manifest pins fetched on 2026-10-16. Python and uv are multi-arch.
FROM ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc AS uv
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

# LocalExecutor samples process-group RSS with ps. Pin the Debian package too.
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps=2:4.0.2-3 libproc2-0=2:4.0.2-3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock LICENSE ./
# Optional Torch uses the same committed lock. No CUDA workload is implied.
ARG TRAINING_TORCH=0
RUN if [ "$TRAINING_TORCH" = 1 ]; then \
      uv sync --locked --no-default-groups --no-install-project --group training-torch; \
    elif [ "$TRAINING_TORCH" = 0 ]; then \
      uv sync --locked --no-default-groups --no-install-project; \
    else exit 2; fi
COPY src/ ./src/
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["python", "-m", "openloop.batch.worker"]
CMD ["--queue", "/queue.json", "--state", "/work", "--sync-to", "/return"]

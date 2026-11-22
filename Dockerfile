FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --locked --no-dev

RUN uv run --no-sync train

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 3333 8501
CMD ["uvicorn", "credit_scoring.api:app", "--host", "0.0.0.0", "--port", "3333"]

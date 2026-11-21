# The trip planner as a container, for Cloud Run Jobs (or `docker run` locally).
# Settings come from environment variables; no .env file is copied in.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

# Run as a normal user; the search cache goes to its home directory.
RUN useradd --create-home planner
USER planner
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
ENTRYPOINT ["travel-plan"]

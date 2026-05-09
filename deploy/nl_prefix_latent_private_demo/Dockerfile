FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

EXPOSE 7860

VOLUME ["/app/experiments/backfill/block_ar/nl_scenario_demo_outputs"]

CMD ["uv", "run", "python", "experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py", "--server-name", "0.0.0.0", "--server-port", "7860", "--require-auth"]

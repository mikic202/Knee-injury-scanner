FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/checkpoints/resnet3d_best_10_01_16:49.pt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock ./
COPY git_submodules/ ./git_submodules/

RUN uv sync --frozen

COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
COPY configs/ ./configs/
COPY README.md .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["PYTHONPATH=.", "uv", "run", "streamlit", "run", "src/web_app/main.py", "--server.address=0.0.0.0"]

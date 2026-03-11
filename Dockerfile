FROM ghcr.io/astral-sh/uv:python3.13-trixie

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

ENV MODEL_SIZE=base
ENV STREAM_URL=https://d.liveatc.net/lszb2_atis
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY main.py ./
CMD ["uv", "run", "python", "/app/main.py"]
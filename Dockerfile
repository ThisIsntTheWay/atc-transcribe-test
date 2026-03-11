FROM ghcr.io/astral-sh/uv:python3.13-trixie

WORKDIR /app

ENV MODEL_SIZE=base
ENV STREAM_URL=https://d.liveatc.net/lszb2_atis

RUN apt update && apt install -y ffmpeg

COPY pyproject.toml main.py ./
RUN uv sync

CMD ["uv", "run", "python", "/app/main.py"]
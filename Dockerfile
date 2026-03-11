FROM python:3.14.3-slim

WORKDIR /app
COPY main.py requirements.txt ./

ENV MODEL_SIZE=base
ENV STREAM_URL=https://d.liveatc.net/lszb2_atis

RUN apt-get update && apt-get install -y ffmpeg && apt-get clean
RUN pip install -r requirements.txt

CMD ["/app/main.py"]
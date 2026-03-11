# ATC/ATIS transcription test

## Deploy
With docker
```bash
docker build . -t atc-transcript:latest

MODEL_SIZE=base # tiny,large
docker run --rm -it -e MODEL_SIZE=$MODEL_SIZE atc-transcript:latest
```

Or, without docker:
```bash
sudo apt install ffmpeg -y
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### Env vars
- `MODEL_SIZE`
  - Can be set to any of the [whisper models](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages).
- `STREAM_URL`
  - Can be set to any mp3 stream
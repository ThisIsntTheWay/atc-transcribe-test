import ffmpeg
import numpy as np
import time
import sys
import logging
from faster_whisper import WhisperModel
import os
import datetime
import uuid
import scipy.io.wavfile

# --- Configuration ---
STREAM_URL = os.getenv("STREAM_URL", "https://d.liveatc.net/lszb2_atis")
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "recordings")
# 10 seconds of audio buffer (16k samples * 4 bytes/sample * 10s)
CHUNK_SIZE = 16000 * 4 * 10 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ATC_Monitor")

# --- Initialize Model ---
logger.info(f"Loading Faster Whisper model: {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
logger.info("Model loaded successfully.")

# --- Create output directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

def start_stream():
    """Starts the FFmpeg process to pipe audio."""
    return (
        ffmpeg
        .input(STREAM_URL, reconnect=1, reconnect_streamed=1, reconnect_delay_max=5)
        .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='16k')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

logger.info(f"Connecting to stream: {STREAM_URL}")
process = start_stream()

while True:
    try:
        # Read raw bytes
        in_bytes = process.stdout.read(CHUNK_SIZE)

        if not in_bytes:
            logger.warning("Stream link broken or empty. Reconnecting in 5s...")
            time.sleep(5)
            process.terminate()
            process = start_stream()
            continue

        # Convert to numpy
        audio_chunk = np.frombuffer(in_bytes, np.float32)

        # Log start of processing
        logger.debug("Starting transcription on audio chunk...")

        # Transcribe
        segments, _ = model.transcribe(
            audio_chunk,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700),
            no_speech_threshold=0.6,
            language="en",
            temperature=0.2,
            initial_prompt="Bern Airport, LSZB, ATIS, Information, QNH, runway"
        )

        # post-process
        transcript_text = ""
        for segment in segments:
            if segment.text.strip():
                transcript_text += segment.text.strip() + " "
        
        transcript_text = transcript_text.strip()

        # Store audio + transcript
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = f"{timestamp}_{str(uuid.uuid4()).split('-')[0]}"
        wav_filename = os.path.join(OUTPUT_DIR, f"{file_id}.wav")
        txt_filename = os.path.join(OUTPUT_DIR, f"{file_id}.txt")

        scipy.io.wavfile.write(wav_filename, 16000, audio_chunk)
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        if transcript_text:
            print(f">>> {transcript_text}")
        else:
            logger.info(f"Heartbeat: Monitoring... (Silence/Static detected) - Saved as {file_id}")

    except Exception as e:
        logger.error(f"Error during processing loop: {e}")
        # Reset stream on error to be safe
        process.terminate()
        time.sleep(2)
        process = start_stream()

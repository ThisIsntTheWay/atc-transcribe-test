import ffmpeg
import numpy as np
import time
import sys
import logging
from faster_whisper import WhisperModel

# --- Configuration ---
STREAM_URL = "https://d.liveatc.net/lszb2_atis"
MODEL_SIZE = "base"
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
            initial_prompt="Bern Airport, LSZB, ATIS, Information, QNH, runway"
        )

        # Process results
        found_speech = False
        for segment in segments:
            if segment.text.strip():
                # Actual transcription log
                print(f">>> {segment.text.strip()}")
                found_speech = True

        if not found_speech:
            logger.info("Heartbeat: Monitoring... (Silence/Static detected)")

    except Exception as e:
        logger.error(f"Error during processing loop: {e}")
        # Reset stream on error to be safe
        process.terminate()
        time.sleep(2)
        process = start_stream()

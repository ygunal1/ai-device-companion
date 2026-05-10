import base64
import time
import tempfile
import os
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import argparse
import signal
import sys

MODEL_NAME = os.getenv("FASTER_WHISPER_MODEL_SIZE_OR_PATH", "tiny")
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))

app = Flask(__name__)

print("[INIT] Loading whisper model...")
t0 = time.perf_counter()
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)
print(f"[INIT] Model loaded in {round(time.perf_counter() - t0, 2)}s")


@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    file_path = data.get("filePath")
    b64_audio = data.get("base64")
    language = data.get("language") or "en"

    if not file_path and not b64_audio:
        return jsonify({"error": "Either filePath or base64 must be provided"}), 400

    temp_file = None
    try:
        if b64_audio:
            fd, temp_file = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with open(temp_file, "wb") as f:
                f.write(base64.b64decode(b64_audio))
            audio_path = temp_file
        else:
            audio_path = file_path

        t0 = time.perf_counter()
        segments, info = model.transcribe(audio_path, language=language, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()

        return jsonify({
            "recognition": text,
            "language": info.language,
            "time_cost": round(time.perf_counter() - t0, 3),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8803)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    print(f"[STARTING] Whisper server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=False)

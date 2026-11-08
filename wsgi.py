"""
WSGI entry point — for Gunicorn / uWSGI.

  gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 600

`python wsgi.py` also loads every model at start (transcription + Ollama) and,
on Ctrl+C or SIGTERM, stops the engine and unloads the Ollama models.
"""
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.app import app  # noqa: E402

if __name__ == "__main__":
    from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
    from workers.engine import get_engine

    engine = get_engine()
    engine.start()  # loads in the background; the dashboard shows the progress
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # same clean-up as Ctrl+C
    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
    finally:
        print("\nStopping the engine and unloading the Ollama models...", flush=True)
        engine.shutdown()

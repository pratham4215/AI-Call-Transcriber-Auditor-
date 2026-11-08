"""
Reviewer dashboard: human reviewers read audited calls and write reviews.

  python reviewer_wsgi.py        (http://<DGX>:5001 by default)

Separate from wsgi.py: it loads no models, has no CRM database access and
can't fetch, transcribe, audit or run batches. It needs no GPU, so it can stay
up while the main dashboard restarts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reviewer.app import app  # noqa: E402

if __name__ == "__main__":
    from config import REVIEWER_HOST, REVIEWER_PORT

    app.run(host=REVIEWER_HOST, port=REVIEWER_PORT, debug=False, threaded=True)

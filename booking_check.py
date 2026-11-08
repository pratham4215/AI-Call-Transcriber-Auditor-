"""TEMPORARY diagnostic (delete after use): why audited Booking calls came out fake.

Run on the DGX from ~/kataria-audit:  python booking_check.py
Reads reports/evidence/*.json and transcripts/; no LLM, no database.
"""
import json
from collections import Counter
from pathlib import Path

from core.audit import pipeline as P

rows = []
for ev_path in sorted(Path("reports/evidence").glob("*.json"), key=lambda p: p.stat().st_mtime):
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    if ev.get("call_type") != "BOOKING" or not isinstance(ev.get("booking"), dict):
        continue
    stem = ev_path.stem
    tx_path = Path("transcripts") / f"{stem}.transcript.json"
    turns = P.parse_transcript(P.read_transcript(tx_path)) if tx_path.exists() else []
    by_id = {f"T{t.index:03d}": t for t in turns}
    stance, stance_turn = P._final_booking_stance(turns)
    b = ev["booking"]
    rows.append({
        "brio": "confidence" in b,
        "genuine": b.get("genuine") is True,
        "reason": b.get("reason"),
        "conf": b.get("confidence"),
        "stance": stance,
        "stance_turn": f"T{stance_turn.index:03d}" if stance_turn else "",
        "date_regex": P._confirmed_booking_fact_turn(turns, P._BOOKING_DATE_RE) is not None,
        "cited": [(tid, by_id[tid].speaker if tid in by_id else "missing") for tid in b.get("turns", [])],
        "evidence": " ".join(str(b.get("evidence") or "").split())[:220],
        "turns": len(turns),
    })

for mode in (False, True):
    group = [r for r in rows if r["brio"] == mode]
    if not group:
        continue
    fake = [r for r in group if not r["genuine"]]
    print(f"\n== {'Brio' if mode else 'generate (before Brio)'}: {len(group)} Booking audits, "
          f"{len(fake)} fake ({100 * len(fake) / len(group):.0f}%)")
    print("  fake reasons:", Counter(r["reason"] for r in fake).most_common())
    print("  Python stance on fakes:", Counter(r["stance"] for r in fake).most_common())
    print("  Python found a date on fakes:", Counter(r["date_regex"] for r in fake).most_common())

print("\n== latest 25 fake Booking audits")
for r in [r for r in rows if not r["genuine"]][-25:]:
    print(f"- {'brio' if r['brio'] else 'gen '} conf={r['conf']} reason={r['reason']} "
          f"stance={r['stance']}@{r['stance_turn']} date_regex={r['date_regex']} cited={r['cited']} turns={r['turns']}")
    print(f"    {r['evidence']}")

"""
One day's audited calls with their human reviews, as a zip (run on the DGX):

  python -m reviewer.export_day 2026-10-19            -> exports/reviews-2026-10-19.zip

Inside: summary.xlsx (every audited call of the day, each parameter, transcript
flags) and calls/<call>/ with the AI audit, every review version, the call
details and the transcript. Same rules as the reviewer dashboard: customer
phone numbers and personal details are removed, and recordings are left out
(a recording carries the customer's voice and spoken numbers).
"""
import argparse
import io
import json
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BASE, REPORTS_DIR as REPORTS  # noqa: E402
from core import reviews  # noqa: E402
from core.audit.excel_report import read_all_scores  # noqa: E402
from core.audit.params import SCORE_OPTIONS  # noqa: E402
from reviewer import app as dash  # noqa: E402

README = """Kataria call reviews for {day}

summary.xlsx
  Calls             every audited call of the day: AI score, review status and the reviewer's verdicts
  Parameters        one row per call and parameter: AI rating vs the reviewer's
  Transcript flags  transcript lines a reviewer marked as transcribed wrongly

calls/<call>/
  audit.json         the AI's ratings and evidence
  review.json        every saved version of the human review, newest first (empty when not reviewed)
  call_details.json  vehicle, service, disposition and booking details the audit used
  transcript.txt     the call transcript, turn by turn (T001 ...)

Customer phone numbers and personal details (name, address, email, PIN code)
are removed. Recordings are not included.
"""


def _safe(value):
    """No spreadsheet formulas from typed text."""
    return "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value


def _rating(table: str, pid: str, points) -> str:
    options = SCORE_OPTIONS.get(table, {}).get(pid, {})
    return next((name for name, value in options.items() if float(value) == float(points)), "Zero")


def _transcript_text(turns: list) -> str:
    lines = []
    for t in turns:
        if t["tid"] is None:
            lines.append(t["text"])
        else:
            role = "CRE" if t["role"] == "Agent" else "Customer"
            minutes, seconds = divmod(int(t["start"]), 60)
            lines.append(f"{t['tid']}  {minutes}:{seconds:02d}  {role}: {t['text']}")
    return "\n".join(lines) + "\n"


def export_day(day: str, out_dir: Path) -> tuple[Path, int, int]:
    """Write the zip; returns (path, calls, reviewed calls)."""
    from openpyxl import Workbook

    scores = read_all_scores(REPORTS)
    rows = [row for row in dash._audited_calls() if row["call_date"] == day]
    book = Workbook()
    calls_sheet = book.active
    calls_sheet.title = "Calls"
    calls_sheet.append(["Call", "Registration", "Time", "Call type", "CRE", "Workshop", "AI score %", "AI grade",
                        "AI unsure (to check)", "Review status", "Reviewer", "Reviewed at", "Agree", "Disagree",
                        "Booking verdict", "Booking genuine (reviewer)", "Call type verdict", "Correct call type",
                        "Overall comment", "Coaching"])
    params_sheet = book.create_sheet("Parameters")
    params_sheet.append(["Call", "Parameter", "AI rating", "AI points", "Max points", "AI source", "AI confidence",
                         "Reviewer verdict", "Reviewer rating", "Note"])
    flags_sheet = book.create_sheet("Transcript flags")
    flags_sheet.append(["Call", "Turn", "Reviewer note"])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"reviews-{day}.zip"
    reviewed = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for row in rows:
            stem = row["stem"]
            score = scores.get(stem) or {}
            table = score.get("table")
            evidence = dash.strip_personal(dash._evidence(stem))
            ev_params = evidence.get("params") or {}
            ai = {pid: {"rating": _rating(table, pid, pts), "points": pts, "max": mx,
                        **{k: ev_params.get(pid, {}).get(k) for k in ("source", "confidence", "check", "evidence", "turns")
                           if ev_params.get(pid, {}).get(k) is not None}}
                  for pid, (pts, mx) in (score.get("params") or {}).items()}
            versions = reviews.history(stem)
            latest = versions[0] if versions else {}
            reviewed += bool(latest)

            folder = f"calls/{stem}"
            zf.writestr(f"{folder}/audit.json", json.dumps({
                "call": stem, "call_type": table, "score_pct": row["score_pct"], "grade": row["grade"],
                "parameters": ai, "booking": evidence.get("booking"),
            }, ensure_ascii=False, indent=1, default=str))
            zf.writestr(f"{folder}/review.json", json.dumps(versions, ensure_ascii=False, indent=1, default=str))
            zf.writestr(f"{folder}/call_details.json", json.dumps(dash._crm(stem), ensure_ascii=False, indent=1, default=str))
            zf.writestr(f"{folder}/transcript.txt", _transcript_text(dash._turns(stem)))

            marks = latest.get("params") or {}
            booking, call_type = latest.get("booking") or {}, latest.get("call_type") or {}
            calls_sheet.append([_safe(v) for v in (
                stem, row["registration"], row["call_time"], row["call_type"],
                " ".join(filter(None, [row["cre_name"], f"({row['cre_code']})" if row["cre_code"] else ""])),
                row["workshop"], row["score_pct"], row["grade"], row["to_check"],
                {"reviewed": "Reviewed", "in_review": "In review"}.get(latest.get("status"), "Not reviewed"),
                latest.get("reviewer"),
                datetime.fromtimestamp(latest["created_at"]).strftime("%Y-%m-%d %H:%M") if latest else None,
                sum(m.get("verdict") == "agree" for m in marks.values()) if latest else None,
                sum(m.get("verdict") == "disagree" for m in marks.values()) if latest else None,
                booking.get("verdict"), {True: "Yes", False: "No"}.get(booking.get("genuine")),
                call_type.get("verdict"), call_type.get("correct"), latest.get("overall"), latest.get("coaching"),
            )])
            for pid, item in ai.items():
                mark = marks.get(pid) or {}
                params_sheet.append([_safe(v) for v in (
                    stem, pid, item["rating"], item["points"], item["max"], item.get("source"),
                    round(item["confidence"], 3) if isinstance(item.get("confidence"), (int, float)) else None,
                    mark.get("verdict"), mark.get("rating"), mark.get("note"),
                )])
            for flag in latest.get("asr_flags") or []:
                flags_sheet.append([_safe(v) for v in (stem, flag.get("turn"), flag.get("note"))])

        buffer = io.BytesIO()
        book.save(buffer)
        zf.writestr("summary.xlsx", buffer.getvalue())
        zf.writestr("README.txt", README.format(day=day))
    return path, len(rows), reviewed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("day", help="call date, YYYY-MM-DD")
    parser.add_argument("--out", default=str(BASE / "exports"), help="folder for the zip (default: exports/)")
    args = parser.parse_args()
    try:
        day = date.fromisoformat(args.day).isoformat()
    except ValueError:
        raise SystemExit("day must be YYYY-MM-DD")
    path, calls, reviewed = export_day(day, Path(args.out))
    print(f"{path}  ({calls} audited calls, {reviewed} reviewed, {path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

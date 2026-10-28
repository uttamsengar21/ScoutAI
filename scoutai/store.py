"""seen.json doubles as the dedupe index AND the application tracker.

Terminology (important — do not conflate these):
  seen      The job has been fetched and recorded at least once.
             Seeing a job does not imply the candidate applied.
  screened  The job has been sent to the LLM screener and has a score.
  emailed   The job appeared in an emailed digest (score >= threshold).
  applied   The candidate manually submitted an application via `scoutai applied <id>`.

A job is never marked applied automatically. Only the human triggers that.

The `unseen()` method filters out jobs that have already been evaluated in the
current configuration. Use `re_evaluate()` to re-surface jobs that were
previously scored below the threshold so they can be re-screened with an
updated prompt or profile, without wiping tracking history.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .fetch import Job


class Store:
    def __init__(self, path: str | Path = "seen.json"):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"  ! {self.path} corrupt — starting fresh (backup written to {self.path}.bak)")
                try:
                    shutil.copy2(self.path, str(self.path) + ".bak")
                except OSError:
                    pass

    # ---------------------------------------------------------------------- #
    #  Deduplication                                                           #
    # ---------------------------------------------------------------------- #

    def unseen(self, jobs: list[Job]) -> list[Job]:
        """Return jobs not yet recorded in the store.

        A job is considered "seen" once it has been recorded with `record()`.
        Jobs that were fetched in a previous run but whose score was below the
        threshold are still "seen" — use `re_evaluate()` to revisit them.
        """
        return [j for j in jobs if j.job_id not in self.data]

    def re_evaluate(self, jobs: list[Job], *, min_rescore_after_days: int = 14) -> list[Job]:
        """Return previously-seen jobs eligible for re-screening.

        A job is returned if it was seen before AND:
          - It has never been screened (score is None/missing), OR
          - It scored below the threshold and was first seen at least
            `min_rescore_after_days` ago (avoids re-screening the same
            rejection in back-to-back runs while still surfacing postings
            that improve under an updated prompt or profile).
          - It has NOT been applied to (applied jobs are frozen).

        The caller is responsible for de-duplicating against `unseen()` — these
        two lists are mutually exclusive by design (unseen has no entry in store;
        re_evaluate only touches entries that ARE in the store).
        """
        cutoff = datetime.now(timezone.utc)
        result: list[Job] = []
        for j in jobs:
            entry = self.data.get(j.job_id)
            if entry is None:
                continue  # not yet seen; let unseen() handle it
            if entry.get("applied"):
                continue  # never re-screen applied jobs

            score = entry.get("score")
            first_seen_str = entry.get("first_seen", "")
            try:
                first_seen = datetime.fromisoformat(first_seen_str)
                if first_seen.tzinfo is None:
                    first_seen = first_seen.replace(tzinfo=timezone.utc)
                days_ago = (cutoff - first_seen).days
            except (ValueError, TypeError):
                days_ago = min_rescore_after_days  # unknown age → allow

            if score is None or days_ago >= min_rescore_after_days:
                result.append(j)

        return result

    # ---------------------------------------------------------------------- #
    #  Recording                                                               #
    # ---------------------------------------------------------------------- #

    def record(self, jobs: list[Job], emailed: bool) -> None:
        """Persist jobs to the store after screening.

        Only sets `first_seen` on the initial write. Subsequent calls update
        `score`, `reason`, and `emailed` without overwriting `first_seen`,
        `applied`, or `applied_on`.

        `emailed` is True only when the mail server confirmed submission. It is
        never set to True merely because the digest was generated locally.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for j in jobs:
            existing = self.data.get(j.job_id)
            if existing is None:
                # First time we see this job — create the full record.
                self.data[j.job_id] = {
                    "first_seen": now,
                    "company": j.company,
                    "title": j.title,
                    "location": j.location,
                    "url": j.url,
                    "score": j.score,
                    "reason": j.reason,
                    "emailed": emailed and (j.score or 0) >= 0,  # track email attempt
                    "applied": False,
                    "applied_on": None,
                }
            else:
                # Job already known — update screening result only.
                existing["score"] = j.score
                existing["reason"] = j.reason
                # Only upgrade emailed flag (never downgrade to False once emailed).
                if emailed:
                    existing["emailed"] = True
        self.save()

    def mark_applied(self, job_id: str) -> bool:
        """Mark a job as applied. Must be triggered by the human, never automatically."""
        if job_id not in self.data:
            return False
        self.data[job_id]["applied"] = True
        self.data[job_id]["applied_on"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.save()
        return True

    # ---------------------------------------------------------------------- #
    #  Reporting                                                               #
    # ---------------------------------------------------------------------- #

    def stats(self) -> dict:
        total = len(self.data)
        screened = sum(1 for v in self.data.values() if v.get("score") is not None)
        emailed = sum(1 for v in self.data.values() if v.get("emailed"))
        applied = sum(1 for v in self.data.values() if v.get("applied"))
        return {
            "tracked": total,
            "screened": screened,
            "emailed": emailed,
            "applied": applied,
        }

    def export_csv(self, path: str | Path = "out/tracker.csv") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["first_seen", "company", "title", "location", "score",
                "reason", "emailed", "applied", "applied_on", "url"]
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["job_id"] + cols, extrasaction="ignore")
            w.writeheader()
            for jid, row in sorted(self.data.items(),
                                   key=lambda kv: kv[1].get("first_seen", ""), reverse=True):
                w.writerow({"job_id": jid, **row})
        return path

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False),
                             encoding="utf-8")

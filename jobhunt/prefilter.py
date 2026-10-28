"""Deterministic filter that runs BEFORE any LLM call.

This is the whole cost story: ~2000 raw jobs -> ~40 candidates for ~0 rupees,
so the LLM only ever reads jobs that already passed title + location + freshness.

Changes from v1:
  - Diagnostic mode: per-reason counts PLUS representative examples of dropped titles/locations.
  - Invalid regex in config now logs a clear error instead of crashing mid-run.
  - Remote-location refinement: allow_remote lets genuine global/India-remote roles through
    but excludes postings that name a specific non-India country ("Remote - US",
    "Remote - Ireland", "Remote - United States", etc.). Truly global or India-explicit remotes still pass.
  - Location matching operates strictly on the location field (or explicit location hints in title like '(Bengaluru, India)'),
    preventing technical title terms like 'Distributed Systems' from leaking through as remote hints.
  - Missing/malformed dates are treated as "unknown age" (not rejected) with a note.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

from .fetch import Job

REMOTE_HINTS = ("remote", "anywhere", "work from home", "wfh")

# Patterns that indicate the remote role is restricted to a non-India country.
_FOREIGN_REMOTE_PAT = re.compile(
    r"\bremote\s*[-–(:]\s*(?!india\b)(?!global\b)(?!worldwide\b)(?!international\b)"
    r"(us\b|usa\b|uk\b|united\s+kingdom|united\s+states|ireland|canada|australia|europe|eu\b|apac|"
    r"latam|emea|mena|singapore|germany|france|netherlands|spain|brazil|japan)",
    re.I,
)

# Explicit non-India locations/countries
_FOREIGN_LOCATION_PAT = re.compile(
    r"\b(united\s+states|usa|san\s+francisco|new\s+york|palo\s+alto|mountain\s+view|seattle|"
    r"bellevue|austin|chicago|london|dublin|ireland|germany|berlin|singapore|toronto|canada|"
    r"tokyo|japan|netherlands|amsterdam|brazil|sao\s+paulo|australia|sydney)\b",
    re.I,
)


def _compile_patterns(patterns: list[str], label: str) -> list[re.Pattern]:
    """Compile regex strings, skipping invalid ones with a printed warning."""
    compiled: list[re.Pattern] = []
    for raw in patterns:
        try:
            compiled.append(re.compile(raw, re.I))
        except re.error as e:
            print(f"  ! config error — invalid {label} regex {raw!r}: {e}")
    return compiled


def _any_match(patterns: list[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    v = value.replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.fromisoformat(v) if fmt is None else datetime.strptime(v, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class _Stats(NamedTuple):
    """Rejection counts plus up-to-5 representative examples per bucket."""
    title: int
    location: int
    age: int
    date_unknown: int  # jobs with no parseable date (passed through, not rejected)
    title_examples: list[str]
    location_examples: list[str]


_MAX_EXAMPLES = 5  # how many rejected titles/locations to show in the summary


def prefilter(jobs: list[Job], cfg: dict, *, verbose: bool = True) -> list[Job]:
    """Return the subset of *jobs* that survive all deterministic filters.

    Parameters
    ----------
    jobs:    raw jobs from all ATS boards
    cfg:     the ``filters:`` sub-dict from config.yaml
    verbose: print a per-reason summary to stdout (default True)
    """
    raw_inc = cfg.get("include_titles") or [r"."]
    raw_exc = cfg.get("exclude_titles") or []
    inc = _compile_patterns(raw_inc, "include_titles")
    exc = _compile_patterns(raw_exc, "exclude_titles")

    locs = [l.lower() for l in (cfg.get("locations") or [])]
    allow_remote = bool(cfg.get("allow_remote", True))
    max_age = cfg.get("max_age_days")
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age) if max_age else None

    kept: list[Job] = []
    counts = {"title": 0, "location": 0, "age": 0, "date_unknown": 0}
    title_ex: list[str] = []
    loc_ex: list[str] = []

    for j in jobs:
        # ---- title gate ----
        if not _any_match(inc, j.title) or (exc and _any_match(exc, j.title)):
            counts["title"] += 1
            if len(title_ex) < _MAX_EXAMPLES:
                title_ex.append(f"{j.title!r} ({j.company})")
            continue

        # ---- location gate ----
        if locs:
            loc_str = (j.location or "").lower()
            
            # Check if title has an explicit location tag like '(Bengaluru, India)' or '- India'
            title_str = (j.title or "").lower()
            combined_loc = f"{loc_str} {title_str}"

            # 1. Does it match an approved Indian city/region?
            matches_india = any(l in loc_str for l in locs) or (
                any(l in title_str for l in locs) and not _FOREIGN_LOCATION_PAT.search(loc_str)
            )

            # 2. Is it a genuine remote role?
            # Check if remote hints exist in the location string
            has_remote_hint = any(h in loc_str for h in REMOTE_HINTS)
            # Or explicit remote tag in title like 'Software Engineer (Remote)'
            if not has_remote_hint:
                has_remote_hint = bool(re.search(r"\b(remote|wfh)\b", title_str))

            is_foreign_remote = bool(_FOREIGN_REMOTE_PAT.search(loc_str)) or (
                _FOREIGN_LOCATION_PAT.search(loc_str) and not any(l in loc_str for l in locs)
            )
            is_valid_remote = allow_remote and has_remote_hint and not is_foreign_remote

            if not matches_india and not is_valid_remote:
                counts["location"] += 1
                if len(loc_ex) < _MAX_EXAMPLES:
                    loc_ex.append(f"{j.location!r} — {j.title!r}")
                continue

        # ---- freshness gate ----
        if cutoff:
            posted = _parse_date(j.posted_at)
            if posted is None:
                counts["date_unknown"] += 1
            elif posted < cutoff:
                counts["age"] += 1
                continue

        kept.append(j)

    if verbose:
        _print_summary(len(jobs), len(kept), counts, title_ex, loc_ex)

    return kept


def _print_summary(total: int, kept: int, counts: dict,
                   title_ex: list[str], loc_ex: list[str]) -> None:
    print(
        f"  prefilter: {total} -> {kept} "
        f"(dropped title={counts['title']} location={counts['location']} "
        f"stale={counts['age']}; {counts['date_unknown']} with unknown date passed through)"
    )
    if title_ex:
        print(f"  rejected-title examples (up to {_MAX_EXAMPLES}):")
        for ex in title_ex:
            print(f"    - {ex}")
    if loc_ex:
        print(f"  rejected-location examples (up to {_MAX_EXAMPLES}):")
        for ex in loc_ex:
            print(f"    - {ex}")

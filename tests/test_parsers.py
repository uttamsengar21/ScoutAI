"""Parsers + prefilter, run against the fixtures in their native ATS shapes.

No network, no API key. This is the suite that catches the two bugs that cost
me an evening each: Lever's epoch-milliseconds timestamps, and a bare `sde`
regex that silently matches nothing.

New in this version:
  - Internship and new-grad title matching tests
  - Backend/full-stack title tests
  - Senior/management exclusion tests
  - India and remote location tests (including foreign-restricted remote)
  - Missing job date handling
  - Invalid regex configuration (graceful skip, no crash)
  - Nested YAML config loading
  - Stable deduplication across repeated runs (Store tests)
  - Re-evaluation of previously seen jobs after config changes
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoutai import mock
from scoutai.fetch import Job, parse_ashby, parse_greenhouse, parse_lever, strip_html
from scoutai.mock import fetch_all_mock
from scoutai.prefilter import prefilter
from scoutai.store import Store

CONFIG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml")
                        .read_text(encoding="utf-8"))
FILTERS = CONFIG["filters"]


# ------------------------------------------------------------- strip_html ---

def test_strip_html_unescapes_twice():
    """Greenhouse ships HTML-entity-escaped HTML: unescape, strip, unescape."""
    raw = "&lt;p&gt;Go &amp;amp; Java&lt;/p&gt;"
    assert strip_html(raw) == "Go & Java"


def test_strip_html_turns_block_tags_into_newlines():
    out = strip_html("<p>One</p><p>Two</p><ul><li>a</li><li>b</li></ul>")
    assert "One" in out and "Two" in out and "a" in out and "b" in out
    assert "<" not in out


def test_strip_html_handles_none_and_empty():
    assert strip_html(None) == ""
    assert strip_html("") == ""


# ---------------------------------------------------------------- parsers ---

def test_greenhouse_maps_every_field():
    jobs = parse_greenhouse("acme-edge", "Acme Edge", mock.GREENHOUSE["acme-edge"])
    j = next(j for j in jobs if j.title.startswith("Software Engineer II"))
    assert j.job_id == "greenhouse:acme-edge:5501001"
    assert j.ats == "greenhouse"
    assert j.company == "Acme Edge"
    assert j.location == "Bangalore, India"
    assert j.url.startswith("https://boards.greenhouse.io/")
    assert "distributed services" in j.description


def test_lever_concatenates_description_lists_and_additional():
    """The requirements live in lists[], not descriptionPlain. Drop the
    concatenation and every Lever job looks unqualified."""
    jobs = parse_lever("quantstack", "QuantStack", mock.LEVER["quantstack"])
    j = next(j for j in jobs if j.title == "Backend Engineer (Go)")
    assert "market data pipeline" in j.description      # descriptionPlain
    assert "Requirements" in j.description              # lists[].text
    assert "2-5 years backend experience" in j.description  # lists[].content
    assert "No take-home" in j.description              # additionalPlain


def test_lever_createdAt_is_epoch_milliseconds():
    """1.7e12 is milliseconds. Reading it as seconds dates the post to 1970
    and the freshness filter eats the whole board without a word."""
    two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).date()
    jobs = parse_lever("quantstack", "QuantStack", mock.LEVER["quantstack"])
    j = next(j for j in jobs if j.title == "Backend Engineer (Go)")
    assert j.posted_at == two_days_ago.isoformat()


def test_ashby_skips_unlisted_drafts():
    jobs = parse_ashby("helioscale", "Helioscale", mock.ASHBY["helioscale"])
    assert all("unlisted" not in j.url for j in jobs)
    assert len(jobs) == 2   # 3 postings, one isListed: false


def test_ashby_reads_compensation_and_html_fallback():
    jobs = parse_ashby("helioscale", "Helioscale", mock.ASHBY["helioscale"])
    networking = next(j for j in jobs if j.title == "Software Engineer, Networking")
    assert networking.salary is not None and "L" in networking.salary
    ds = next(j for j in jobs if j.title == "Data Scientist, Growth")
    assert "Causal inference" in ds.description   # descriptionHtml fallback


def test_job_ids_are_globally_unique_and_namespaced():
    jobs = fetch_all_mock()
    ids = [j.job_id for j in jobs]
    assert len(ids) == len(set(ids))
    assert all(re.match(r"^(greenhouse|lever|ashby):[^:]+:.+$", i) for i in ids)


def test_parsers_take_decoded_json_not_a_response():
    """Parsers are pure: body in, list[Job] out. That is what makes --mock
    exercise the real code path instead of a second implementation."""
    assert parse_greenhouse("x", "X", {}) == []
    assert parse_lever("x", "X", []) == []
    assert parse_ashby("x", "X", {}) == []


# -------------------------------------------------------------- prefilter ---

@pytest.mark.parametrize("title", [
    # Core SDE/SWE titles
    "Software Engineer II, Distributed Systems",
    "Software Development Engineer, Core Infra",
    "Backend Engineer (Go)",
    "Site Reliability Engineer",
    "SDE II",
    # Internship-specific titles (the critical ones for this candidate)
    "Software Engineering Intern",
    "Software Engineer Intern",
    "SWE Intern",
    "SDE Intern",
    "Backend Engineering Intern",
    "Backend Developer Intern",
    "Full Stack Developer Intern",
    "Full-Stack Engineer Intern",
    "Frontend Engineer Intern",
    "Web Developer Intern",
    "Software Development Intern",
    # New-grad/entry-level
    "Graduate Software Engineer",
    "Junior Software Engineer",
    "Associate Software Engineer",
    "New Grad Software Engineer",
    "Entry-Level Software Engineer",
])
def test_include_titles_match_real_titles(title):
    inc_raw = FILTERS["include_titles"]
    assert any(re.search(p, title, re.I) for p in inc_raw), (
        f"Title {title!r} was not matched by any include pattern"
    )


def test_bare_sde_regex_does_not_match_the_spelled_out_title():
    """The bug: `sde` looks like it covers "Software Development Engineer".
    It does not -- they share no substring. \\bsde\\b plus the spelled-out
    variant is why both titles survive the filter."""
    assert not re.search(r"\bsde\b", "Software Development Engineer", re.I)
    assert re.search(r"\bsde\b", "SDE II", re.I)
    inc = FILTERS["include_titles"]
    assert any(re.search(p, "Software Development Engineer, Core Infra", re.I) for p in inc)


@pytest.mark.parametrize("title,reason", [
    ("Staff Software Engineer, Storage", "too senior"),
    ("Engineering Manager, Platform", "management track"),
    ("Enterprise Account Executive", "wrong function/sales"),
    ("Data Scientist, Growth", "wrong discipline"),
    ("Principal Engineer, Infra", "too senior"),
    ("Director of Engineering", "director/management"),
    ("VP Engineering", "VP level"),
])
def test_junk_titles_are_rejected(title, reason):
    """Senior, management, and non-engineering roles must be excluded.

    Note: 'Frontend Engineer, Design Systems' is intentionally NOT in this list
    because the candidate targets frontend internships and React roles.
    'Frontend Engineer' is a valid match -- only UI/non-engineering disciplines
    like 'Product Designer' or 'Graphic Designer' are excluded.
    """
    inc_raw, exc_raw = FILTERS["include_titles"], FILTERS["exclude_titles"]
    included = any(re.search(p, title, re.I) for p in inc_raw)
    excluded = any(re.search(p, title, re.I) for p in exc_raw)
    assert excluded or not included, (
        f"{title!r} ({reason}) would have survived the filter"
    )


def test_frontend_engineer_is_a_valid_target_title():
    """The candidate targets React/Frontend internships -- 'Frontend Engineer'
    must NOT be excluded. Only UI/non-engineering roles are excluded."""
    inc_raw = FILTERS["include_titles"]
    exc_raw = FILTERS["exclude_titles"]
    title = "Frontend Engineer, Design Systems"
    included = any(re.search(p, title, re.I) for p in inc_raw)
    excluded = any(re.search(p, title, re.I) for p in exc_raw)
    assert included and not excluded, (
        "Frontend Engineer should be included -- candidate targets React/frontend roles"
    )


def test_full_mock_funnel_expected_titles():
    """Check that the mock fixtures produce a known-good set of titles.

    Updated to reflect the new intern/new-grad fixtures added to mock.py
    and the correct behaviour for frontend roles.
    """
    kept = prefilter(fetch_all_mock(), FILTERS)
    kept_titles = sorted(j.title for j in kept)
    # Must include
    assert "Software Engineering Intern" in kept_titles, \
        "Internship fixture must pass the filter"
    assert "Backend Engineering Intern" in kept_titles, \
        "Backend intern fixture must pass"
    assert "Graduate Software Engineer" in kept_titles, \
        "New-grad fixture must pass"
    assert "Backend Engineer (Go)" in kept_titles
    assert "Software Development Engineer, Core Infra" in kept_titles
    # Must exclude
    assert "Staff Software Engineer, Storage" not in kept_titles
    assert "Engineering Manager, Platform" not in kept_titles
    assert "Enterprise Account Executive" not in kept_titles
    assert "Data Scientist, Growth" not in kept_titles


def test_stale_posting_is_dropped_by_freshness_gate():
    kept = prefilter(fetch_all_mock(), FILTERS)
    assert not any("Senior Software Engineer, Platform" == j.title for j in kept)


def test_wrong_city_dropped_but_india_remote_kept():
    kept = prefilter(fetch_all_mock(), FILTERS)
    assert not any("San Francisco" in (j.location or "") for j in kept)
    assert any("Remote" in (j.location or "") for j in kept)


def test_allow_remote_is_what_lets_an_out_of_region_remote_role_through():
    """"Remote (India)" already matches the `india` location, so it is the
    wrong fixture for this. Use a remote role that names no allowed city."""
    remote = Job(job_id="lever:x:1", ats="lever", company="X",
                 title="Backend Engineer", location="Remote - Global",
                 url="https://example.com", description="Go")

    kept_on = prefilter([remote], dict(FILTERS, allow_remote=True))
    kept_off = prefilter([remote], dict(FILTERS, allow_remote=False))

    assert len(kept_on) == 1
    assert kept_off == []


def test_foreign_restricted_remote_is_rejected():
    """'Remote - US' and 'Remote - Ireland' must not pass as remote-global.
    These locations are useless for an India-based student intern.
    """
    for loc in ("Remote - US", "Remote - Ireland", "Remote - UK", "Remote - Canada"):
        job = Job(job_id=f"gh:x:{loc}", ats="greenhouse", company="X",
                  title="Software Engineer", location=loc,
                  url="https://example.com", description="Python")
        kept = prefilter([job], FILTERS)
        assert kept == [], f"Foreign-restricted remote {loc!r} should be rejected"


def test_india_explicit_remote_is_kept():
    """'Remote - India' is always accepted."""
    job = Job(job_id="gh:x:1", ats="greenhouse", company="X",
              title="Backend Engineer", location="Remote - India",
              url="https://example.com", description="Node.js")
    kept = prefilter([job], FILTERS)
    assert len(kept) == 1


def test_empty_filters_keep_everything():
    jobs = fetch_all_mock()
    assert len(prefilter(jobs, {})) == len(jobs)


# ------------------------------------------------------- date handling -----

def test_missing_date_does_not_reject_job():
    """A job with no posted_at must pass the freshness filter (unknown age)."""
    job = Job(job_id="gh:x:1", ats="greenhouse", company="X",
              title="Backend Engineer", location="Bangalore",
              url="https://example.com", description="Go", posted_at=None)
    kept = prefilter([job], FILTERS)
    assert len(kept) == 1


def test_malformed_date_does_not_reject_job():
    """An unparseable date string must be treated as unknown, not rejected."""
    job = Job(job_id="gh:x:2", ats="greenhouse", company="X",
              title="Software Engineer", location="Pune",
              url="https://example.com", description="Java", posted_at="not-a-date")
    kept = prefilter([job], FILTERS)
    assert len(kept) == 1


def test_clearly_stale_date_is_rejected():
    """A job posted 400 days ago (well past max_age_days=30) must be dropped."""
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    job = Job(job_id="gh:x:3", ats="greenhouse", company="X",
              title="Software Engineer", location="Hyderabad",
              url="https://example.com", description="Java", posted_at=old)
    kept = prefilter([job], FILTERS)
    assert kept == []


# -------------------------------------------------- invalid regex handling --

def test_invalid_include_regex_is_skipped_gracefully(capsys):
    """A broken regex in config must not crash the run -- it prints a warning
    and the remaining valid patterns are still applied."""
    bad_filters = dict(FILTERS, include_titles=["[invalid", r"\bsoftware\b"])
    job = Job(job_id="gh:x:1", ats="greenhouse", company="X",
              title="Software Engineer", location="Bangalore",
              url="https://example.com", description="")
    kept = prefilter([job], bad_filters)
    out = capsys.readouterr().out
    assert "invalid" in out.lower() or "error" in out.lower()
    # The valid pattern still matches; job should be kept.
    assert len(kept) == 1


def test_invalid_exclude_regex_is_skipped_gracefully(capsys):
    """Broken exclude regex must not crash. Job should still be evaluated."""
    bad_filters = dict(FILTERS, exclude_titles=["[broken", r"\bstaff\b"])
    senior_job = Job(job_id="gh:x:1", ats="greenhouse", company="X",
                     title="Staff Software Engineer", location="Bangalore",
                     url="https://example.com", description="")
    kept = prefilter([senior_job], bad_filters)
    out = capsys.readouterr().out
    assert "invalid" in out.lower() or "error" in out.lower()
    # The valid exclude pattern \bstaff\b still fires; senior job should be rejected.
    assert kept == []


# --------------------------------------------- nested YAML config loading --

def test_filters_are_loaded_as_nested_dict_not_string():
    """config.yaml nests filters under a 'filters:' key. If the whole file
    is passed instead of cfg['filters'], every job either passes or fails
    on the wrong keys. This catches that mistake early."""
    assert isinstance(FILTERS, dict), "FILTERS must be the nested sub-dict"
    assert "include_titles" in FILTERS
    assert "exclude_titles" in FILTERS
    assert "locations" in FILTERS
    assert isinstance(FILTERS["include_titles"], list)


def test_yaml_config_file_loads_all_expected_sections():
    """Regression: ensure top-level config keys are present and typed."""
    assert "filters" in CONFIG
    assert isinstance(CONFIG.get("score_threshold"), (int, float))
    assert isinstance(CONFIG.get("max_per_digest"), int)
    assert isinstance(CONFIG.get("screen_batch_size"), int)


# -------------------------------------------------- Store deduplication ----

def _make_job(n: int, loc: str = "Bangalore") -> Job:
    return Job(
        job_id=f"greenhouse:acme:{n}",
        ats="greenhouse",
        company="Acme",
        title="Software Engineer Intern",
        location=loc,
        url=f"https://example.com/{n}",
        description="React and Node.js internship.",
    )


def test_store_unseen_returns_all_jobs_when_empty(tmp_path):
    store = Store(tmp_path / "seen.json")
    jobs = [_make_job(i) for i in range(3)]
    assert store.unseen(jobs) == jobs


def test_store_unseen_excludes_previously_recorded_jobs(tmp_path):
    store = Store(tmp_path / "seen.json")
    jobs = [_make_job(i) for i in range(3)]
    store.record(jobs[:2], emailed=False)

    new = store.unseen(jobs)
    assert len(new) == 1
    assert new[0].job_id == jobs[2].job_id


def test_store_record_preserves_first_seen_on_re_record(tmp_path):
    """Calling record() twice for the same job must not overwrite first_seen."""
    store = Store(tmp_path / "seen.json")
    j = _make_job(1)
    store.record([j], emailed=False)
    first = store.data[j.job_id]["first_seen"]

    # Simulate a second run: same job, different score.
    j.score = 8.0
    store.record([j], emailed=True)

    assert store.data[j.job_id]["first_seen"] == first, (
        "first_seen must not change on subsequent records"
    )
    assert store.data[j.job_id]["score"] == 8.0


def test_store_applied_is_never_set_by_record(tmp_path):
    """Screening or emailing must never mark a job as applied."""
    store = Store(tmp_path / "seen.json")
    j = _make_job(1)
    j.score = 9.0
    store.record([j], emailed=True)
    assert store.data[j.job_id]["applied"] is False
    assert store.data[j.job_id]["applied_on"] is None


def test_store_mark_applied_sets_flag_and_timestamp(tmp_path):
    store = Store(tmp_path / "seen.json")
    j = _make_job(1)
    store.record([j], emailed=False)
    result = store.mark_applied(j.job_id)
    assert result is True
    entry = store.data[j.job_id]
    assert entry["applied"] is True
    assert entry["applied_on"] is not None


def test_store_mark_applied_returns_false_for_unknown_id(tmp_path):
    store = Store(tmp_path / "seen.json")
    assert store.mark_applied("nonexistent:job:id") is False


def test_store_stats_counts_correctly(tmp_path):
    store = Store(tmp_path / "seen.json")
    jobs = [_make_job(i) for i in range(4)]
    jobs[0].score = 8.0
    jobs[1].score = 7.5
    jobs[2].score = 3.0
    store.record(jobs, emailed=False)
    store.mark_applied(jobs[0].job_id)

    stats = store.stats()
    assert stats["tracked"] == 4
    assert stats["applied"] == 1


def test_store_re_evaluate_surfaces_low_score_jobs_after_cooldown(tmp_path):
    """Jobs that scored below threshold should be eligible for re-screening
    after `min_rescore_after_days` days."""
    store = Store(tmp_path / "seen.json")
    j = _make_job(1)
    j.score = 3.0
    store.record([j], emailed=False)

    # Backdate the first_seen entry to simulate a job seen 20 days ago.
    old_date = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(timespec="seconds")
    store.data[j.job_id]["first_seen"] = old_date
    store.save()

    # Reload from disk to ensure persistence is correct.
    store2 = Store(tmp_path / "seen.json")
    result = store2.re_evaluate([j], min_rescore_after_days=14)
    assert len(result) == 1
    assert result[0].job_id == j.job_id


def test_store_re_evaluate_respects_cooldown(tmp_path):
    """Jobs seen < min_rescore_after_days ago must NOT be re-evaluated."""
    store = Store(tmp_path / "seen.json")
    j = _make_job(1)
    j.score = 3.0
    store.record([j], emailed=False)
    # first_seen is just now (0 days ago)

    result = store.re_evaluate([j], min_rescore_after_days=14)
    assert result == [], "Job seen today must not be re-screened"


def test_store_re_evaluate_skips_applied_jobs(tmp_path):
    """Applied jobs must never be re-evaluated."""
    store = Store(tmp_path / "seen.json")
    j = _make_job(1)
    j.score = 2.0
    store.record([j], emailed=False)

    # Age the entry and mark as applied.
    old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
    store.data[j.job_id]["first_seen"] = old_date
    store.save()
    store.mark_applied(j.job_id)

    store2 = Store(tmp_path / "seen.json")
    result = store2.re_evaluate([j], min_rescore_after_days=14)
    assert result == [], "Applied jobs must never be re-screened"


def test_store_persist_and_reload(tmp_path):
    """Records survive a save+load cycle without corruption."""
    path = tmp_path / "seen.json"
    store = Store(path)
    j = _make_job(42)
    j.score = 8.5
    j.reason = "Great React match."
    store.record([j], emailed=True)

    store2 = Store(path)
    entry = store2.data.get(j.job_id)
    assert entry is not None
    assert entry["score"] == 8.5
    assert entry["reason"] == "Great React match."
    assert entry["emailed"] is True
    assert entry["applied"] is False


def test_store_export_csv_produces_valid_file(tmp_path):
    store = Store(tmp_path / "seen.json")
    j = _make_job(1)
    j.score = 7.0
    store.record([j], emailed=False)
    csv_path = store.export_csv(tmp_path / "tracker.csv")

    import csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["job_id"] == j.job_id
    assert rows[0]["score"] == "7.0"

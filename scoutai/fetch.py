"""Fetch jobs from public ATS APIs. No auth, no scraping, no ToS risk."""
from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Iterable

import requests

UA = {"User-Agent": "scoutai/1.0 (autonomous career search agent)"}
TIMEOUT = 20

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")


def strip_html(raw: str | None) -> str:
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"<\s*(br|/p|/div|/li|/h[1-6])\s*/?>", "\n", text, flags=re.I)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()


@dataclass
class Job:
    job_id: str          # stable global id for dedupe: "<ats>:<slug>:<id>"
    ats: str
    company: str
    title: str
    location: str
    url: str
    description: str
    posted_at: str | None = None
    salary: str | None = None
    # filled in later by the pipeline
    score: float | None = None
    reason: str | None = None
    draft: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Adapters. Each takes the raw JSON body and returns list[Job].
# Keeping parse separate from HTTP is what makes offline testing possible.
# --------------------------------------------------------------------------

def parse_greenhouse(slug: str, company: str, body: Any) -> list[Job]:
    out = []
    for j in (body or {}).get("jobs", []):
        loc = (j.get("location") or {}).get("name") or ""
        out.append(Job(
            job_id=f"greenhouse:{slug}:{j.get('id')}",
            ats="greenhouse",
            company=company,
            title=(j.get("title") or "").strip(),
            location=loc.strip(),
            url=j.get("absolute_url") or "",
            description=strip_html(j.get("content")),
            posted_at=j.get("updated_at") or j.get("first_published"),
        ))
    return out


def parse_lever(slug: str, company: str, body: Any) -> list[Job]:
    out = []
    for j in (body or []):
        cats = j.get("categories") or {}
        # Lever splits the JD across descriptionPlain + a `lists` array.
        chunks = [j.get("descriptionPlain") or strip_html(j.get("description"))]
        for lst in (j.get("lists") or []):
            chunks.append(str(lst.get("text") or ""))
            chunks.append(strip_html(lst.get("content")))
        chunks.append(j.get("additionalPlain") or strip_html(j.get("additional")))
        ts = j.get("createdAt")
        posted = None
        if isinstance(ts, (int, float)):
            posted = time.strftime("%Y-%m-%d", time.gmtime(ts / 1000))
        out.append(Job(
            job_id=f"lever:{slug}:{j.get('id')}",
            ats="lever",
            company=company,
            title=(j.get("text") or "").strip(),
            location=(cats.get("location") or "").strip(),
            url=j.get("hostedUrl") or j.get("applyUrl") or "",
            description="\n\n".join(c for c in chunks if c).strip(),
            posted_at=posted,
            salary=cats.get("commitment"),
        ))
    return out


def parse_ashby(slug: str, company: str, body: Any) -> list[Job]:
    out = []
    for j in (body or {}).get("jobs", []):
        if j.get("isListed") is False:
            continue
        comp = j.get("compensation") or {}
        salary = None
        summary = comp.get("compensationTierSummary") or comp.get("summaryComponents")
        if isinstance(summary, str):
            salary = summary
        out.append(Job(
            job_id=f"ashby:{slug}:{j.get('id')}",
            ats="ashby",
            company=company,
            title=(j.get("title") or "").strip(),
            location=(j.get("location") or "").strip(),
            url=j.get("jobUrl") or j.get("applyUrl") or "",
            description=(j.get("descriptionPlain") or strip_html(j.get("descriptionHtml")) or "").strip(),
            posted_at=j.get("publishedAt"),
            salary=salary,
        ))
    return out


def parse_workday(tenant: str, wd: int | str, site: str, company: str, body: Any) -> list[Job]:
    out = []
    base_url = f"https://{tenant}.wd{wd}.myworkdayjobs.com/en-US/{site}"
    for j in (body or {}).get("jobPostings", []):
        ext_path = j.get("externalPath") or ""
        job_code = ext_path.split("_")[-1] if "_" in ext_path else ext_path
        out.append(Job(
            job_id=f"workday:{tenant}:{job_code}",
            ats="workday",
            company=company,
            title=(j.get("title") or "").strip(),
            location=(j.get("locationsText") or "").strip(),
            url=f"{base_url}{ext_path}",
            description=(j.get("title") or "") + " " + (j.get("locationsText") or ""),
            posted_at=j.get("postedOn"),
        ))
    return out


ENDPOINTS = {
    "greenhouse": ("https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true", parse_greenhouse),
    "lever":      ("https://api.lever.co/v0/postings/{slug}?mode=json", parse_lever),
    "ashby":      ("https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true", parse_ashby),
}


def fetch_board(ats: str, slug: str | None = None, company: str | None = None,
                session: requests.Session | None = None,
                config: dict | None = None) -> list[Job]:
    """Hit one company's public board. Returns [] on any failure (never raises)."""
    cfg = config or {}
    sess = session or requests

    if ats == "workday":
        tenant = cfg.get("tenant") or slug or ""
        wd = cfg.get("wd", 1)
        site = cfg.get("site", "External")
        api_url = f"https://{tenant}.wd{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        all_jobs = []
        try:
            for page in range(2):
                payload = {"appliedFacets": {}, "limit": 20, "offset": page * 20, "searchText": ""}
                r = sess.post(api_url, json=payload, headers=headers, timeout=TIMEOUT)
                if r.status_code != 200:
                    break
                page_jobs = parse_workday(tenant, wd, site, company or tenant, r.json())
                all_jobs.extend(page_jobs)
                if len(page_jobs) < 20:
                    break
            return all_jobs
        except Exception as e:
            print(f"  ! workday/{tenant} -> {type(e).__name__}: {e}")
            return []

    if ats not in ENDPOINTS:
        raise ValueError(f"unknown ATS: {ats}")
    url_tpl, parser = ENDPOINTS[ats]
    try:
        r = sess.get(url_tpl.format(slug=slug), headers=UA, timeout=TIMEOUT)
        if r.status_code != 200:
            print(f"  ! {ats}/{slug} -> HTTP {r.status_code}")
            return []
        return parser(slug or "", company or slug or "", r.json())
    except Exception as e:  # dead slug, rate limit, network blip
        print(f"  ! {ats}/{slug} -> {type(e).__name__}: {e}")
        return []


def fetch_all(companies: Iterable[dict], sleep: float = 0.25) -> list[Job]:
    jobs: list[Job] = []
    session = requests.Session()
    for c in companies:
        slug = c.get("slug") or c.get("tenant") or ""
        got = fetch_board(c["ats"], slug, c.get("name"), session=session, config=c)
        if got:
            print(f"  {c.get('name') or slug:<28} {len(got):>4} jobs  ({c['ats']})")
        jobs.extend(got)
        time.sleep(sleep)
    return jobs

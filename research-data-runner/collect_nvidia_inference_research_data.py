#!/usr/bin/env python3
import csv, json, os, re, time, heapq
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import requests

SNAPSHOT = "2026-10-21"
REPOS = [
    ("NVIDIA", "TensorRT-LLM", "tensorrt-llm"),
    ("NVIDIA", "TensorRT", "tensorrt"),
    ("triton-inference-server", "server", "triton-server"),
]
OUT = Path(os.environ.get("DATA_OUT", "data-out"))
TOKEN = os.environ.get("GITHUB_TOKEN", "")
API = "https://api.github.com"
S = requests.Session()
S.headers.update({
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "jiah-research-data-collector/1.0",
})
if TOKEN:
    S.headers["Authorization"] = "Bearer " + TOKEN

def request_json(method, url, **kwargs):
    for attempt in range(8):
        r = S.request(method, url, timeout=90, **kwargs)
        if r.status_code in (403, 429):
            reset = r.headers.get("X-RateLimit-Reset")
            retry = r.headers.get("Retry-After")
            wait = 60
            if retry:
                try: wait = max(2, int(retry))
                except: pass
            elif reset:
                try: wait = max(2, int(reset) - int(time.time()) + 2)
                except: pass
            print(f"rate-limited {r.status_code}; sleep {wait}s")
            time.sleep(min(wait, 180))
            continue
        if r.status_code >= 500:
            time.sleep(min(2 ** attempt, 30))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GitHub request failed after retries: {url}")

def paginate(url, params=None, max_pages=1000):
    params = dict(params or {})
    page = 1
    while page <= max_pages:
        p = dict(params)
        p.update({"per_page": 100, "page": page})
        arr = request_json("GET", url, params=p)
        if not isinstance(arr, list):
            raise TypeError(f"Expected list from {url}, got {type(arr)}")
        if not arr:
            break
        for x in arr:
            yield x
        if len(arr) < 100:
            break
        page += 1
        time.sleep(0.08)

def iso_dt(s):
    if not s: return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def hours(a, b):
    if not a or not b: return None
    return round((iso_dt(b) - iso_dt(a)).total_seconds() / 3600, 6)

def issue_num_from_url(url):
    if not url: return None
    m = re.search(r"/issues/(\d+)$", url)
    return int(m.group(1)) if m else None

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n")

def compact_issue(x, repo):
    return {
        "repo": repo, "number": x.get("number"), "id": x.get("id"), "node_id": x.get("node_id"),
        "title": x.get("title"), "body": x.get("body"),
        "state": x.get("state"), "state_reason": x.get("state_reason"), "locked": x.get("locked"),
        "author_association": x.get("author_association"), "user_type": (x.get("user") or {}).get("type"),
        "labels": [z if isinstance(z, str) else z.get("name") for z in (x.get("labels") or [])],
        "assignee_count": len(x.get("assignees") or []),
        "milestone": None if not x.get("milestone") else {
            "number": x["milestone"].get("number"), "title": x["milestone"].get("title"),
            "state": x["milestone"].get("state")
        },
        "comments_count": x.get("comments") or 0,
        "reactions_total": (x.get("reactions") or {}).get("total_count") or 0,
        "created_at": x.get("created_at"), "updated_at": x.get("updated_at"),
        "closed_at": x.get("closed_at"), "html_url": x.get("html_url")
    }

def compact_comment(x, repo):
    n = issue_num_from_url(x.get("issue_url"))
    return {
        "repo": repo, "issue_number": n, "id": x.get("id"),
        "created_at": x.get("created_at"), "updated_at": x.get("updated_at"),
        "author_association": x.get("author_association"),
        "user_type": (x.get("user") or {}).get("type"),
        "body": x.get("body"), "html_url": x.get("html_url")
    }

def compact_event(x, repo):
    issue = x.get("issue") or {}
    n = issue.get("number")
    if n is None:
        n = issue_num_from_url(issue.get("url"))
    return {
        "repo": repo, "issue_number": n, "id": x.get("id"), "event": x.get("event"),
        "created_at": x.get("created_at"), "actor_type": (x.get("actor") or {}).get("type"),
        "label": (x.get("label") or {}).get("name"),
        "assignee_type": (x.get("assignee") or {}).get("type"),
        "milestone": (x.get("milestone") or {}).get("title"),
        "commit_id": x.get("commit_id"),
        "rename_from": (x.get("rename") or {}).get("from"),
        "rename_to": (x.get("rename") or {}).get("to"),
        "lock_reason": x.get("lock_reason")
    }

def compact_pull(x, repo):
    return {
        "repo": repo, "number": x.get("number"), "id": x.get("id"), "title": x.get("title"),
        "body": x.get("body"), "state": x.get("state"), "draft": x.get("draft"),
        "author_association": x.get("author_association"), "user_type": (x.get("user") or {}).get("type"),
        "labels": [z if isinstance(z, str) else z.get("name") for z in (x.get("labels") or [])],
        "created_at": x.get("created_at"), "updated_at": x.get("updated_at"),
        "closed_at": x.get("closed_at"), "merged_at": x.get("merged_at"),
        "merge_commit_sha": x.get("merge_commit_sha"), "html_url": x.get("html_url")
    }

def compact_release(x, repo):
    return {
        "repo": repo, "id": x.get("id"), "tag_name": x.get("tag_name"), "name": x.get("name"),
        "draft": x.get("draft"), "prerelease": x.get("prerelease"),
        "created_at": x.get("created_at"), "published_at": x.get("published_at"),
        "target_commitish": x.get("target_commitish"), "html_url": x.get("html_url")
    }

def has_nontrivial_section(body, terms):
    if not body: return 0
    txt = body.replace("\r", "")
    lines = txt.split("\n")
    for i, line in enumerate(lines):
        clean = re.sub(r"[#*_:\-]", " ", line.lower()).strip()
        if any(t in clean for t in terms):
            buf = []
            for j in range(i+1, min(i+12, len(lines))):
                if re.match(r"^\s*#{1,6}\s+", lines[j]):
                    break
                buf.append(lines[j])
            content = re.sub(r"[-*\[\]xX<>#_]", " ", "\n".join(buf))
            content = re.sub(r"\s+", " ", content).strip()
            if len(content) >= 20:
                return 1
    return 0

def info_proxy(body):
    env = has_nontrivial_section(body, ["system info", "environment", "system information"])
    repro = has_nontrivial_section(body, ["reproduction", "steps to reproduce", "steps to repro", "reproduce"])
    observed = has_nontrivial_section(body, ["actual behavior", "actual result", "error", "logs"])
    expected = has_nontrivial_section(body, ["expected behavior", "expected result"])
    vals = [env, repro, observed, expected]
    return env, repro, observed, expected, round(sum(vals)/4, 2)

STRICT_STAFF = {"OWNER", "MEMBER", "COLLABORATOR"}
LOOSE_STAFF = STRICT_STAFF | {"CONTRIBUTOR"}

def graph_issue_edits(owner, name, out_path):
    query = """
    query($owner:String!, $name:String!, $cursor:String) {
      repository(owner:$owner, name:$name) {
        issues(first:100, after:$cursor, orderBy:{field:CREATED_AT,direction:ASC}) {
          pageInfo { hasNextPage endCursor }
          nodes {
            number
            userContentEdits(first:100) {
              totalCount
              nodes { id editedAt deletedAt diff }
            }
          }
        }
      }
    }
    """
    cursor = None
    checked = 0
    edited = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        while True:
            payload = {"query": query, "variables": {"owner": owner, "name": name, "cursor": cursor}}
            r = S.post("https://api.github.com/graphql", json=payload, timeout=120)
            if r.status_code != 200:
                return {"ok": False, "status": r.status_code, "text": r.text[:1000], "checked": checked, "edited": edited}
            j = r.json()
            if j.get("errors"):
                return {"ok": False, "errors": j["errors"], "checked": checked, "edited": edited}
            block = j["data"]["repository"]["issues"]
            for node in block["nodes"]:
                checked += 1
                edits = node.get("userContentEdits") or {}
                if (edits.get("totalCount") or 0) > 0:
                    edited += 1
                    f.write(json.dumps({
                        "repo": owner + "/" + name,
                        "issue_number": node["number"],
                        "edit_total_count": edits.get("totalCount"),
                        "edits": edits.get("nodes") or []
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
            if not block["pageInfo"]["hasNextPage"]:
                break
            cursor = block["pageInfo"]["endCursor"]
            time.sleep(0.15)
    return {"ok": True, "checked": checked, "edited": edited}

def derive_analysis(issues, comments, events, repo):
    comments_by = defaultdict(list)
    for c in comments:
        if c["issue_number"] is not None:
            comments_by[c["issue_number"]].append(c)
    for v in comments_by.values():
        v.sort(key=lambda x: x.get("created_at") or "")

    event_counts = defaultdict(lambda: defaultdict(int))
    for e in events:
        n = e.get("issue_number")
        if n is not None:
            event_counts[n][e.get("event") or ""] += 1

    external = [x for x in issues if x.get("author_association") not in STRICT_STAFF]
    external_sorted = sorted(external, key=lambda x: x["created_at"] or "")
    load_map = {}
    for days in (7, 14, 30):
        q = deque()
        for x in external_sorted:
            t = iso_dt(x["created_at"])
            cutoff = t.timestamp() - days*86400
            while q and q[0] < cutoff:
                q.popleft()
            load_map[(x["number"], days)] = len(q)
            q.append(t.timestamp())

    backlog = {}
    close_heap = []
    for x in external_sorted:
        t = iso_dt(x["created_at"]).timestamp()
        while close_heap and close_heap[0] <= t:
            heapq.heappop(close_heap)
        backlog[x["number"]] = len(close_heap)
        if x.get("closed_at"):
            heapq.heappush(close_heap, iso_dt(x["closed_at"]).timestamp())
        else:
            heapq.heappush(close_heap, float("inf"))

    rows = []
    for x in issues:
        comms = comments_by.get(x["number"], [])
        first_any = comms[0]["created_at"] if comms else None
        strict = next((c for c in comms if c.get("author_association") in STRICT_STAFF and c.get("user_type") != "Bot"), None)
        loose = next((c for c in comms if c.get("author_association") in LOOSE_STAFF and c.get("user_type") != "Bot"), None)
        env, repro, obs, exp, info = info_proxy(x.get("body"))
        is_external = 1 if x.get("author_association") not in STRICT_STAFF else 0
        row = {
            "repo": repo, "number": x["number"], "state": x.get("state"), "state_reason": x.get("state_reason"),
            "author_association": x.get("author_association"), "is_external_author": is_external,
            "created_at": x.get("created_at"), "closed_at": x.get("closed_at"),
            "first_comment_at": first_any,
            "first_staff_response_strict_at": strict.get("created_at") if strict else None,
            "first_staff_response_loose_at": loose.get("created_at") if loose else None,
            "created_to_first_staff_hours": hours(x.get("created_at"), strict.get("created_at") if strict else None),
            "first_staff_to_close_hours": hours(strict.get("created_at") if strict else None, x.get("closed_at")),
            "created_to_close_hours": hours(x.get("created_at"), x.get("closed_at")),
            "right_censored": 1 if not x.get("closed_at") else 0,
            "comments_count_api": x.get("comments_count"), "comments_rows": len(comms),
            "label_count": len(x.get("labels") or []),
            "info_environment_proxy": env, "info_reproduction_proxy": repro,
            "info_observed_proxy": obs, "info_expected_proxy": exp,
            "info_completeness_proxy_current_body": info,
            "load_external_7d": load_map.get((x["number"],7)) if is_external else None,
            "load_external_14d": load_map.get((x["number"],14)) if is_external else None,
            "load_external_30d": load_map.get((x["number"],30)) if is_external else None,
            "backlog_external_at_creation": backlog.get(x["number"]) if is_external else None,
            "event_labeled_count": event_counts[x["number"]].get("labeled",0),
            "event_unlabeled_count": event_counts[x["number"]].get("unlabeled",0),
            "event_reopened_count": event_counts[x["number"]].get("reopened",0),
            "event_closed_count": event_counts[x["number"]].get("closed",0),
            "event_assigned_count": event_counts[x["number"]].get("assigned",0),
        }
        rows.append(row)
    return rows

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def collect_repo(owner, name, slug):
    repo = owner + "/" + name
    root = OUT / slug
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print("COLLECT", repo)

    meta = request_json("GET", f"{API}/repos/{repo}")
    (root/"repo_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    issues = []
    pr_count_seen = 0
    for x in paginate(f"{API}/repos/{repo}/issues", {"state":"all","sort":"created","direction":"asc"}):
        if x.get("pull_request"):
            pr_count_seen += 1
            continue
        issues.append(compact_issue(x, repo))
    issue_nums = {x["number"] for x in issues}
    write_jsonl(raw/"issues.jsonl", issues)

    comments = []
    for x in paginate(f"{API}/repos/{repo}/issues/comments", {"sort":"created","direction":"asc"}):
        c = compact_comment(x, repo)
        if c["issue_number"] in issue_nums:
            comments.append(c)
    write_jsonl(raw/"comments.jsonl", comments)

    events = []
    for x in paginate(f"{API}/repos/{repo}/issues/events"):
        e = compact_event(x, repo)
        if e["issue_number"] in issue_nums:
            events.append(e)
    write_jsonl(raw/"events.jsonl", events)

    pulls = [compact_pull(x, repo) for x in paginate(f"{API}/repos/{repo}/pulls", {"state":"all","sort":"created","direction":"asc"})]
    write_jsonl(raw/"pulls.jsonl", pulls)

    releases = [compact_release(x, repo) for x in paginate(f"{API}/repos/{repo}/releases")]
    write_jsonl(raw/"releases.jsonl", releases)

    tags = list(paginate(f"{API}/repos/{repo}/tags"))
    write_jsonl(raw/"tags.jsonl", tags)

    labels = list(paginate(f"{API}/repos/{repo}/labels"))
    write_jsonl(raw/"labels.jsonl", labels)

    milestones = list(paginate(f"{API}/repos/{repo}/milestones", {"state":"all"}))
    write_jsonl(raw/"milestones.jsonl", milestones)

    edit_diag = graph_issue_edits(owner, name, raw/"issue_edits.jsonl")

    analysis = derive_analysis(issues, comments, events, repo)
    write_csv(root/"analysis_ready.csv", analysis)

    open_n = sum(1 for x in issues if x.get("state")=="open")
    closed_n = sum(1 for x in issues if x.get("state")=="closed")
    external_n = sum(1 for x in issues if x.get("author_association") not in STRICT_STAFF)
    strict_resp_n = sum(1 for x in analysis if x.get("first_staff_response_strict_at"))
    null_body = sum(1 for x in issues if not x.get("body"))
    info_hist = defaultdict(int)
    for x in analysis:
        info_hist[str(x["info_completeness_proxy_current_body"])] += 1

    diag = {
        "repository": repo, "snapshot_date": SNAPSHOT,
        "issues": len(issues), "open_issues": open_n, "closed_issues": closed_n,
        "external_author_issues": external_n, "prs_seen_and_excluded_from_issue_endpoint": pr_count_seen,
        "comments_on_true_issues": len(comments), "events_on_true_issues": len(events),
        "pull_requests": len(pulls), "releases": len(releases), "tags": len(tags),
        "labels": len(labels), "milestones": len(milestones),
        "null_or_empty_issue_bodies": null_body,
        "strict_staff_first_response_observed": strict_resp_n,
        "strict_staff_response_coverage_all_issues": round(strict_resp_n/len(issues),4) if issues else None,
        "information_completeness_proxy_distribution": dict(sorted(info_hist.items())),
        "issue_edit_history_collection": edit_diag,
        "date_min_created": min((x["created_at"] for x in issues if x.get("created_at")), default=None),
        "date_max_created": max((x["created_at"] for x in issues if x.get("created_at")), default=None),
        "notes": [
            "Initial-information completeness proxy uses the CURRENT REST issue body and is exploratory only.",
            "issue_edits.jsonl stores GraphQL userContentEdits where available so creation-time body reconstruction can be validated later.",
            "Strict maintainer response = first non-bot comment with author_association OWNER, MEMBER, or COLLABORATOR.",
            "External author = issue author_association not OWNER/MEMBER/COLLABORATOR.",
            "Analysis-ready workload counts use all external true issues, not yet the final technical-support-only sample."
        ]
    }
    (root/"data_quality.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    return diag

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_diag = []
    for owner, name, slug in REPOS:
        all_diag.append(collect_repo(owner, name, slug))
    manifest = {
        "snapshot_date": SNAPSHOT,
        "repositories": [x[0]+"/"+x[1] for x in REPOS],
        "diagnostics": all_diag,
        "research_stages": {
            "KCI": "TensorRT-LLM workload/capacity saturation and completion performance",
            "Masters": "TensorRT-LLM stage-specific first-response vs post-response bottlenecks plus information-completeness moderation",
            "SCI_SCIE": "Cross-repository replication across TensorRT-LLM, TensorRT, and Triton Inference Server"
        }
    }
    (OUT/"collection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT/"README.txt").write_text(
        "NVIDIA AI inference support research data snapshot " + SNAPSHOT + "\n"
        "Contains compact raw JSONL plus analysis-ready CSVs for three repositories.\n"
        "Public GitHub data. See each data_quality.json for limitations and coverage.\n",
        encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

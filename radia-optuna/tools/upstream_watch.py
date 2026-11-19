"""Report whether official Optuna has a stable release newer than the pin.

Read-only. Queries the GitHub REST API of optuna/optuna for tag names, the
commits of two tags and the files that differ between them. When the release
descends from the pin, the comparison lists the files (with renames). When it
does not (diverged history), the merge-base comparison would miss changes
made only on the pinned side, so the file list is the exact difference of the
two tag trees instead. It never clones, downloads or runs upstream code, and
changes nothing in this repository.

Usage:
    python tools/upstream_watch.py [--report-md PATH] [--report-json PATH]

When GITHUB_STEP_SUMMARY is set, the Markdown report is appended to it.
GITHUB_TOKEN, when set, is sent only to api.github.com to raise the rate limit.

Exit status (each outcome is distinct; none of them means "no update" except 0):
    0  up_to_date        the pin is the latest stable release
    1  update_available  a newer stable release exists; the report lists its changes
    3  pin_tag_missing   no official tag matches the pinned version
    4  incomplete        the file list is partial or missing; the report says so
    5  error             network, HTTP or response-format failure

The scheduled workflow maps 1 to a warning on a successful run and fails on
3, 4 and 5 (see .github/workflows/radia-optuna-upstream-watch.yml).
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

UPSTREAM = "optuna/optuna"
API = f"https://api.github.com/repos/{UPSTREAM}"
WEB = f"https://github.com/{UPSTREAM}"
PIN_PATH = Path(__file__).resolve().parents[1] / "src/radia_optuna/upstream.json"
STABLE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
# GitHub lists at most this many files for one comparison; a full page may be truncated.
COMPARE_FILE_LIMIT = 300
EXIT = {"up_to_date": 0, "update_available": 1, "pin_tag_missing": 3, "incomplete": 4, "error": 5}

# First match wins. Prefixes end with "/" for directories.
COMPONENTS = (
    ("tpe", ("optuna/samplers/_tpe/",)),
    ("pruners", ("optuna/pruners/",)),
    ("storage", ("optuna/storages/",)),
    ("core_api", ("optuna/study/", "optuna/trial/", "optuna/search_space/", "optuna/distributions.py",
                  "optuna/_transform.py", "optuna/samplers/_base.py", "optuna/samplers/_random.py",
                  "optuna/samplers/_lazy_random_state.py", "optuna/samplers/__init__.py",
                  "optuna/exceptions.py", "optuna/logging.py", "optuna/_experimental.py",
                  "optuna/_deprecated.py", "optuna/__init__.py", "optuna/version.py")),
    ("other_samplers", ("optuna/samplers/",)),
    ("other_api", ("optuna/",)),
    ("tests", ("tests/",)),
    ("build_ci", (".github/", "pyproject.toml", "setup.py", "setup.cfg", "requirements", "MANIFEST.in")),
    ("docs", ("docs/", "tutorial/", "README", "CONTRIBUTING", "CHANGELOG")),
)
TITLES = {
    "tpe": "TPE sampler", "pruners": "Pruners", "storage": "Storage",
    "core_api": "Core API (study, trial, distributions, search space, base/random samplers)",
    "other_samplers": "Other samplers", "other_api": "Other optuna modules",
    "tests": "Upstream tests", "build_ci": "Build and CI", "docs": "Documentation", "other": "Other paths",
}
# Inferred, not observed: which radia-optuna checks a change in each component calls for.
REVIEW = {
    "tpe": "TPE numerics or random consumption may change: regenerate the upstream and CUDA density "
           "oracles and recheck CPU/GPU TPE parity.",
    "pruners": "Check the MATLAB pruner classes and their oracle cases.",
    "storage": "Check tests/test_storage_bridge.py and the table/HDF5 storage extension semantics.",
    "core_api": "Check study/trial/distribution semantics, search-space order, warnings, the public API "
                "inventory (oracles/optuna_public_api.json) and matlab/optuna_api_coverage.json.",
    "other_samplers": "Check whether the changed samplers are mirrored in the MATLAB package.",
    "other_api": "Check the public API inventory for added or removed names.",
    "tests": "Upstream tests changed: regenerate oracles/optuna_test_manifest.json in the move.",
}


class WatchError(Exception):
    """Network, HTTP or response-format failure."""


def stable_version(tag):
    match = STABLE_TAG.fullmatch(tag)
    return tuple(int(part) for part in match.groups()) if match else None


def release_base(tag):
    """X.Y.Z of any vX.Y.Z... tag, including prereleases such as v5.2.0rc1."""
    match = re.match(r"^v(\d+)\.(\d+)\.(\d+)(?!\d)", tag)
    return tuple(int(part) for part in match.groups()) if match else None


def classify(path):
    for component, prefixes in COMPONENTS:
        if any(path == p or path.startswith(p) for p in prefixes):
            return component
    return "other"


def http_get(url, token=None):
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "radia-optuna-upstream-watch"})
    if token and urllib.parse.urlsplit(url).hostname == "api.github.com":
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8")), response.headers.get("Link", "")
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise WatchError(f"GET {url}: {exc}") from exc


class GitHub:
    """The read-only queries the watcher needs."""

    def __init__(self, token=None, get=http_get):
        self.token, self.get = token, get

    def tags(self):
        names, url = [], f"{API}/tags?per_page=100"
        while url:
            page, link = self.get(url, self.token)
            if not isinstance(page, list):
                raise WatchError(f"unexpected tag list: {page!r:.200}")
            names += [item["name"] for item in page]
            url = next((part.split(";")[0].strip(" <>") for part in link.split(",")
                        if 'rel="next"' in part), None)
        return names

    def commit(self, tag):
        """Commit and root-tree SHAs of a tag."""
        data, _ = self.get(f"{API}/commits/{urllib.parse.quote(tag)}", self.token)
        sha = data.get("sha") if isinstance(data, dict) else None
        tree = ((data.get("commit") or {}).get("tree") or {}).get("sha") if isinstance(data, dict) else None
        if not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{40}", v) for v in (sha, tree)):
            raise WatchError(f"tag {tag} did not resolve to a commit and tree")
        return {"tag": tag, "sha": sha, "tree": tree}

    def tree(self, sha):
        """Every non-directory entry of a tree as {path: (mode, object SHA)}, and the truncation flag."""
        data, _ = self.get(f"{API}/git/trees/{sha}?recursive=1", self.token)
        if not isinstance(data, dict) or not isinstance(data.get("tree"), list):
            raise WatchError(f"unexpected tree response for {sha}")
        entries = {e["path"]: (e["mode"], e["sha"]) for e in data["tree"] if e["type"] != "tree"}
        return entries, bool(data.get("truncated"))

    def compare(self, base, head):
        data, _ = self.get(f"{API}/compare/{base}...{head}", self.token)
        if not isinstance(data, dict):
            raise WatchError(f"unexpected comparison response: {data!r:.200}")
        return data


def changes(files):
    rows = []
    for item in files:
        row = {"path": item["filename"], "status": item["status"], "component": classify(item["filename"])}
        if item.get("previous_filename"):
            row["previous_path"] = item["previous_filename"]
            row["previous_component"] = classify(item["previous_filename"])
        rows.append(row)
    return rows


def tree_delta(old, new):
    """Exact path difference of two trees. Git trees carry no renames: a move is a removal plus an addition."""
    rows = []
    for path in sorted(old.keys() | new.keys()):
        if path not in new:
            status = "removed"
        elif path not in old:
            status = "added"
        elif old[path] != new[path]:
            status = "modified"
        else:
            continue
        rows.append({"filename": path, "status": status})
    return rows


def watch(pin, github):
    report = {"schema": "radia-optuna.upstream-watch.v1", "upstream": UPSTREAM, "pin": pin}
    try:
        pinned = stable_version(f"v{pin}")
        if pinned is None:
            report.update(status="pin_tag_missing", detail=f"pin {pin!r} is not a stable X.Y.Z version")
            return report
        tags = github.tags()
        stable = sorted((stable_version(t), t) for t in tags if stable_version(t))
        # Prereleases of versions above the pin, listed so that their exclusion is visible.
        report["excluded_prerelease_tags"] = sorted(
            (t for t in tags if not stable_version(t) and (release_base(t) or ()) > pinned),
            key=lambda t: (release_base(t), t))
        if f"v{pin}" not in tags:
            report.update(status="pin_tag_missing", detail=f"no official tag v{pin} in {UPSTREAM}")
            return report
        latest_version, latest = stable[-1]
        report["pinned_tag"] = github.commit(f"v{pin}")
        if latest_version <= pinned:
            report["latest_stable_tag"] = report["pinned_tag"]
            report["status"] = "up_to_date"
            return report
        report["latest_stable_tag"] = github.commit(latest)
        report["newer_stable_tags"] = [t for v, t in stable if v > pinned]
        base, head = report["pinned_tag"]["sha"], report["latest_stable_tag"]["sha"]
        comparison = github.compare(base, head)
        report.update(
            compare_url=f"{WEB}/compare/v{pin}...{latest}",
            release_url=f"{WEB}/releases/tag/{latest}",
            comparison_status=comparison.get("status"),
            ahead_by=comparison.get("ahead_by"), behind_by=comparison.get("behind_by"))
        partial = None
        if comparison.get("status") == "ahead":
            # The pin is an ancestor: the comparison is the whole pin-to-release change.
            report["file_source"] = "comparison"
            files = comparison.get("files")
            if not isinstance(files, list):
                files, partial = [], "the comparison returned no file list"  # omitted for very large comparisons
            elif len(files) >= COMPARE_FILE_LIMIT:
                partial = f"the comparison lists {len(files)} files, GitHub's limit"
        else:
            # Diverged: the merge-base comparison omits pin-only changes. Use the exact tree difference.
            report["file_source"] = "tree_difference"
            old, old_cut = github.tree(report["pinned_tag"]["tree"])
            new, new_cut = github.tree(report["latest_stable_tag"]["tree"])
            files = tree_delta(old, new)
            if old_cut or new_cut:
                partial = "GitHub truncated a recursive tree listing"
        report["files"] = changes(files)
        report["files_partial"] = partial is not None
        groups = {}
        for row in report["files"]:
            for key in {row["component"], row.get("previous_component", row["component"])}:
                groups.setdefault(key, []).append(row["path"])
        report["components"] = {key: len(paths) for key, paths in sorted(groups.items())}
        report["review_points"] = [REVIEW[key] for key, _ in COMPONENTS if key in groups and key in REVIEW]
        if partial:
            report.update(status="incomplete", detail=f"{partial}; the changed-file list below is partial and may "
                          f"miss affected components. Review {report['compare_url']} manually.")
        else:
            report["status"] = "update_available"
        return report
    except (WatchError, KeyError, TypeError) as exc:
        report.update(status="error", detail=str(exc))
        return report


def markdown(report):
    lines = [f"# Upstream Optuna watch: `{report['status']}`", "",
             f"Pin: `{report['pin']}` (`radia-optuna/src/radia_optuna/upstream.json`)."]
    if "detail" in report:
        lines += ["", f"**{report['status']}:** {report['detail']}"]
    for key, label in (("pinned_tag", "Pinned"), ("latest_stable_tag", "Latest stable")):
        if key in report:
            lines.append(f"- {label}: `{report[key]['tag']}` = `{report[key]['sha']}`")
    if report.get("excluded_prerelease_tags"):
        lines.append("- Excluded prereleases: " + ", ".join(f"`{t}`" for t in report["excluded_prerelease_tags"]))
    if "files" in report:
        lines += [f"- Newer stable tags: {', '.join(report['newer_stable_tags'])}",
                  f"- Compare: {report['compare_url']}", f"- Release notes: {report['release_url']}",
                  f"- Comparison status: `{report['comparison_status']}` (ahead {report['ahead_by']}, "
                  f"behind {report['behind_by']})"]
        if report["file_source"] == "tree_difference":
            lines.append("  The release does not descend from the pin. The file list is the exact difference "
                         "of the two tag trees; a moved file shows as removed plus added.")
        heading = "partial: not every changed file is listed" if report["files_partial"] else "observed"
        lines += ["", f"## Changed files ({heading})", ""]
        for key in [c for c, _ in COMPONENTS] + ["other"]:
            rows = [r for r in report["files"] if key in (r["component"], r.get("previous_component"))]
            if not rows:
                continue
            lines.append(f"### {TITLES[key]} ({len(rows)})")
            for r in rows:
                moved = f" (from `{r['previous_path']}`)" if "previous_path" in r else ""
                lines.append(f"- {r['status']}: `{r['path']}`{moved}")
            lines.append("")
        lines += ["## Review points (inferred from paths, not from behaviour)", ""]
        lines += [f"- {point}" for point in report["review_points"]] or ["- None beyond the release notes."]
        lines += ["- Moving the pin follows the optuna-upstream-follow skill and needs the user's agreement."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))["optuna"]
    report = watch(pin, GitHub(os.environ.get("GITHUB_TOKEN")))
    text = markdown(report)
    sys.stdout.write(text)
    if args.report_md:
        args.report_md.write_text(text, encoding="utf-8", newline="\n")
    if args.report_json:
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(text)
    return EXIT[report["status"]]


if __name__ == "__main__":
    sys.exit(main())

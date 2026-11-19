"""Guards: the upstream watcher tells no update, a newer stable release and every failure apart.

Assumes: tools/upstream_watch.py reads only the GitHub REST responses faked
here (tag list, tag commit, comparison). No network is used.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/upstream_watch.py"
spec = importlib.util.spec_from_file_location("upstream_watch", SCRIPT)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)

API = watch.API
SHA = {tag: f"{i:040x}" for i, tag in enumerate(["v9.0.0", "v9.1.0", "v9.1.1", "v9.2.0"], start=1)}


class FakeGet:
    """Serve canned responses keyed by URL; any other URL is a network failure."""

    def __init__(self, tags, files=None, compare=None, pages=None):
        self.responses = {f"{API}/commits/{tag}": ({"sha": sha}, "") for tag, sha in SHA.items()}
        if pages:
            urls = [f"{API}/tags?per_page=100"] + [f"{API}/tags?page={n}" for n in range(2, len(pages) + 1)]
            for n, (url, page) in enumerate(zip(urls, pages)):
                link = f'<{urls[n + 1]}>; rel="next"' if n + 1 < len(urls) else ""
                self.responses[url] = ([{"name": t} for t in page], link)
        else:
            self.responses[f"{API}/tags?per_page=100"] = ([{"name": t} for t in tags], "")
        body = compare if compare is not None else {"status": "ahead", "ahead_by": 3, "behind_by": 0, "files": files or []}
        for base in SHA.values():
            for head in SHA.values():
                self.responses[f"{API}/compare/{base}...{head}"] = (body, "")
        self.urls = []

    def __call__(self, url, token):
        self.urls.append(url)
        if url not in self.responses:
            raise watch.WatchError(f"GET {url}: <urlopen error [Errno 11001] getaddrinfo failed>")
        return self.responses[url]


def run(pin, get):
    report = watch.watch(pin, watch.GitHub(get=get))
    json.dumps(report)  # the report must always serialize
    return report


def test_no_update_when_pin_is_latest_stable():
    get = FakeGet(["v9.1.0", "v9.0.0", "v8.9.0"])
    report = run("9.1.0", get)
    assert (report["status"], watch.EXIT[report["status"]]) == ("up_to_date", 0), report
    assert report["pinned_tag"] == report["latest_stable_tag"] == {"tag": "v9.1.0", "sha": SHA["v9.1.0"]}
    assert not any("/compare/" in url for url in get.urls), get.urls
    assert "files" not in report


def test_newer_stable_release_is_reported_with_tag_sha_pairs_and_links():
    files = [{"filename": "optuna/samplers/_tpe/sampler.py", "status": "modified"}]
    report = run("9.1.0", FakeGet(["v9.0.0", "v9.1.0", "v9.1.1", "v9.2.0"], files))
    assert (report["status"], watch.EXIT[report["status"]]) == ("update_available", 1), report
    assert report["pinned_tag"] == {"tag": "v9.1.0", "sha": SHA["v9.1.0"]}
    assert report["latest_stable_tag"] == {"tag": "v9.2.0", "sha": SHA["v9.2.0"]}
    assert report["newer_stable_tags"] == ["v9.1.1", "v9.2.0"]
    assert report["compare_url"] == "https://github.com/optuna/optuna/compare/v9.1.0...v9.2.0"
    text = watch.markdown(report)
    assert "`v9.2.0` = `" + SHA["v9.2.0"] + "`" in text and report["compare_url"] in text, text


def test_version_order_is_numeric_not_lexical():
    get = FakeGet(["v9.9.0", "v9.10.0"])
    get.responses.update({f"{API}/commits/{t}": ({"sha": c * 40}, "") for t, c in (("v9.9.0", "9"), ("v9.10.0", "a"))})
    get.responses[f"{API}/compare/{'9' * 40}...{'a' * 40}"] = ({"status": "ahead", "files": []}, "")
    report = run("9.9.0", get)
    assert (report["status"], report["latest_stable_tag"]["tag"]) == ("update_available", "v9.10.0"), report


def test_prereleases_and_nonrelease_tags_are_excluded_and_listed():
    tags = ["v9.1.0", "v9.2.0rc1", "v9.2.0b1", "v10.0.0a1", "v9.1.0rc1", "nightly", "9.3.0"]
    report = run("9.1.0", FakeGet(tags))
    assert report["status"] == "up_to_date", report
    assert report["excluded_prerelease_tags"] == ["v9.2.0b1", "v9.2.0rc1", "v10.0.0a1"], report


def test_component_classification_covers_tpe_pruners_storage_core_and_renames():
    files = [
        {"filename": "optuna/samplers/_tpe/parzen_estimator.py", "status": "modified"},
        {"filename": "optuna/pruners/_median.py", "status": "added"},
        {"filename": "optuna/storages/_rdb/storage.py", "status": "removed"},
        {"filename": "optuna/distributions.py", "status": "modified"},
        {"filename": "optuna/samplers/_cmaes.py", "status": "modified"},
        {"filename": "optuna/visualization/_plot.py", "status": "modified"},
        {"filename": "tests/samplers_tests/test_tpe.py", "status": "modified"},
        {"filename": "docs/source/index.rst", "status": "modified"},
        {"filename": "optuna/storages/journal/_file.py", "status": "renamed",
         "previous_filename": "optuna/samplers/_tpe/_old.py"},
    ]
    report = run("9.1.0", FakeGet(["v9.1.0", "v9.2.0"], files))
    by_path = {row["path"]: row for row in report["files"]}
    assert {p: r["component"] for p, r in by_path.items()} == {
        "optuna/samplers/_tpe/parzen_estimator.py": "tpe", "optuna/pruners/_median.py": "pruners",
        "optuna/storages/_rdb/storage.py": "storage", "optuna/distributions.py": "core_api",
        "optuna/samplers/_cmaes.py": "other_samplers", "optuna/visualization/_plot.py": "other_api",
        "tests/samplers_tests/test_tpe.py": "tests", "docs/source/index.rst": "docs",
        "optuna/storages/journal/_file.py": "storage"}
    renamed = by_path["optuna/storages/journal/_file.py"]
    assert (renamed["previous_path"], renamed["previous_component"]) == ("optuna/samplers/_tpe/_old.py", "tpe")
    # A rename counts in both components and produces both review points.
    assert report["components"]["tpe"] == 2 and report["components"]["storage"] == 2, report["components"]
    assert [p for p in report["review_points"] if p in (watch.REVIEW["tpe"], watch.REVIEW["storage"])] == [
        watch.REVIEW["tpe"], watch.REVIEW["storage"]]
    text = watch.markdown(report)
    for status in ("added", "removed", "renamed"):
        assert f"- {status}: `" in text, text
    assert "(from `optuna/samplers/_tpe/_old.py`)" in text
    assert "inferred from paths, not from behaviour" in text and "Changed files (observed)" in text


def test_diverged_release_is_flagged_but_not_incomplete():
    compare = {"status": "diverged", "ahead_by": 5, "behind_by": 2, "files": []}
    report = run("9.1.0", FakeGet(["v9.1.0", "v9.2.0"], compare=compare))
    assert report["status"] == "update_available", report
    assert "relative to their merge base" in watch.markdown(report)


def test_tag_pagination_is_followed():
    report = run("9.1.0", FakeGet(None, pages=[["v9.1.0", "v9.0.0"], ["v9.2.0"]]))
    assert report["latest_stable_tag"]["tag"] == "v9.2.0", report


@pytest.mark.parametrize("pin, tags, detail", [
    ("9.1.0", ["v9.0.0", "v9.2.0"], "no official tag v9.1.0"),
    ("9.1.0rc1", ["v9.1.0"], "not a stable X.Y.Z version"),
])
def test_missing_pin_tag_is_not_no_update(pin, tags, detail):
    report = run(pin, FakeGet(tags))
    assert (report["status"], watch.EXIT[report["status"]]) == ("pin_tag_missing", 3), report
    assert detail in report["detail"], report


def test_truncated_or_missing_file_list_is_incomplete():
    truncated = [{"filename": f"optuna/f{i}.py", "status": "modified"} for i in range(watch.COMPARE_FILE_LIMIT)]
    for compare in ({"status": "ahead", "files": truncated}, {"status": "ahead"}):
        report = run("9.1.0", FakeGet(["v9.1.0", "v9.2.0"], compare=compare))
        assert (report["status"], watch.EXIT[report["status"]]) == ("incomplete", 4), report
        assert "manually" in report["detail"] and report["compare_url"] in report["detail"], report


@pytest.mark.parametrize("get", [
    lambda url, token: (_ for _ in ()).throw(watch.WatchError(f"GET {url}: <urlopen error timed out>")),
    lambda url, token: ({"message": "API rate limit exceeded"}, ""),
])
def test_network_or_format_failure_is_error_not_no_update(get):
    report = run("9.1.0", get)
    assert (report["status"], watch.EXIT[report["status"]]) == ("error", 5), report
    assert report["detail"], report
    assert "error" in watch.markdown(report)


def test_unresolvable_tag_commit_is_error():
    get = FakeGet(["v9.1.0", "v9.2.0"])
    get.responses[f"{API}/commits/v9.2.0"] = ({"message": "Not Found"}, "")
    report = run("9.1.0", get)
    assert report["status"] == "error" and "v9.2.0" in report["detail"], report


def test_token_goes_only_to_the_github_api(monkeypatch):
    seen = []

    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"[]"

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, request.get_header("Authorization")))
        return Response()

    monkeypatch.setattr(watch.urllib.request, "urlopen", fake_urlopen)
    watch.http_get(f"{API}/tags", "secret")
    watch.http_get("https://example.invalid/api.github.com/tags", "secret")
    assert seen == [(f"{API}/tags", "Bearer secret"), ("https://example.invalid/api.github.com/tags", None)], seen


def test_main_writes_reports_and_summary_with_distinct_exit(tmp_path, monkeypatch):
    real = watch.GitHub
    monkeypatch.setattr(watch, "GitHub", lambda token: real(get=FakeGet(["v0.0.1"])))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    code = watch.main(["--report-md", str(tmp_path / "r.md"), "--report-json", str(tmp_path / "r.json")])
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert (code, report["status"]) == (3, "pin_tag_missing"), report
    assert summary.read_text(encoding="utf-8") == (tmp_path / "r.md").read_text(encoding="utf-8")

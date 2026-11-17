"""Guard the active upstream pin and the identities of frozen inputs."""

import json
from pathlib import Path
import re
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = "src/radia_optuna/upstream.json"
PIN = json.loads((ROOT / PIN_PATH).read_text(encoding="utf-8"))

# These are hash-bound production evidence or recorded historical provenance.
FROZEN = {
    "CHANGELOG.md",
    "optuna_handover.md",
    "optuna_review.md",
    "oracles/README.md",
    "oracles/generate_optuna_api_inventory.py",
    "oracles/generate_optuna50_cuda_oracle.py",
    "oracles/generate_optuna_mcp_oracle.py",
    "oracles/generate_optuna_oracle.py",
    "oracles/generate_optuna_cuda_density_oracle.py",
    "oracles/optuna50_cuda_oracle.json",
    "oracles/optuna_mcp_oracle.json",
    "oracles/optuna_oracle.json",
    "oracles/optuna_public_api.json",
    "oracles/optuna_cuda_density_oracle.json",
    "oracles/optuna_test_manifest.json",
    "matlab/optuna_api_coverage.json",
    "matlab/optuna_upstream_compatibility.json",
    "src/radia_optuna/evidence_manifest.json",
}


def test_upstream_pin_matches_compatibility_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = project["project"]["optional-dependencies"]["upstream"]
    assert [req for req in extra if req.startswith("optuna==")] == [f"optuna=={PIN['optuna']}"]
    assert "upstream.json" in project["tool"]["setuptools"]["package-data"]["radia_optuna"]
    for name in ("manifest.json", "evidence_manifest.json"):
        record = json.loads((ROOT / "src/radia_optuna" / name).read_text(encoding="utf-8"))
        assert record["oracle_version"] == PIN["optuna"], name
    for key, name in (("upstream_oracle", "optuna_oracle.json"),
                      ("cuda_oracle", "optuna50_cuda_oracle.json")):
        fixture = json.loads((ROOT / "oracles" / name).read_text(encoding="utf-8"))
        for dependency, version in PIN["generation"][key].items():
            assert fixture[f"{dependency}_version"] == version, (name, dependency)
    mcp = json.loads((ROOT / "oracles/optuna_mcp_oracle.json").read_text(encoding="utf-8"))
    assert mcp["optuna_mcp_version"] == PIN["optuna_mcp"]


def test_no_active_upstream_optuna_pin_literals():
    # Check version tokens, including regex-escaped tokens. The optuna-mcp
    # version alone can collide with unrelated source versions (e.g. schemas).
    files = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    git_root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True
    ).strip())
    if git_root == ROOT.parent and ROOT.name == "radia-optuna":
        files.extend(f"../.github/workflows/{name}" for name in (
            "radia-optuna.yml", "release-radia-optuna.yml"
        ))
    allow = FROZEN | {PIN_PATH, "pyproject.toml", "src/radia_optuna/manifest.json"}
    violations = []
    for file in sorted(set(files) - allow):
        path = ROOT / file
        if path.suffix not in {".md", ".json", ".py", ".m", ".yml", ".yaml", ".ps1", ".toml", ".cpp", ".cu", ".hpp"}:
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.replace(r"\\.", ".").replace(r"\.", ".")
            mcp = re.search(r"optuna[-_]mcp.{0,12}" + re.escape(PIN["optuna_mcp"]), line, re.I)
            if PIN["optuna"] in line or mcp:
                violations.append(f"{file}:{line_no}")
    assert not violations, violations

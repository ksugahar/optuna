"""Regenerate Optuna reference fixtures in a new directory outside the checkout."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from zipfile import ZipFile


PRODUCERS = (
    ("generate_optuna_oracle.py", "optuna50_oracle.json", "optuna_oracle.json"),
    ("generate_optuna_mcp_oracle.py", "optuna50_mcp_oracle.json", "optuna_mcp_oracle.json"),
    ("generate_optuna_api_inventory.py", "optuna50_public_api.json", "optuna_public_api.json"),
    ("generate_optuna50_cuda_oracle.py", "optuna50_cuda_oracle.json", "optuna50_cuda_oracle.json"),
    ("generate_optuna_cuda_density_oracle.py", "optuna_cuda_density_oracle.json", "optuna_cuda_density_oracle.json"),
)
PIN_PATH = Path("src/radia_optuna/upstream.json")
COVERAGE = Path("matlab/optuna_api_coverage.json")
MANIFEST = Path("oracles/optuna_test_manifest.json")
CUDA_PRODUCERS = {producer for producer, _, _ in PRODUCERS[-2:]}


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args])


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def report_value(value):
    """Normalize non-finite numbers only in the report, leaving fixtures intact."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": json.dumps(value)}
    if isinstance(value, dict):
        return {key: report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [report_value(item) for item in value]
    return value


def require_python(pin):
    required = pin["generation"]["upstream_oracle"]["python"]
    actual = platform.python_version()
    if actual != required:
        raise ValueError(f"Oracle generation requires Python {required}, found {actual}")


def changes(before, after, path=()):
    """Yield changes at JSON paths, retaining sequence order and RNG values."""
    if isinstance(before, dict) and isinstance(after, dict):
        for key in dict.fromkeys((*before, *after)):
            location = (*path, key)
            if key not in before:
                yield {"path": location, "kind": "added", "after": after[key]}
            elif key not in after:
                yield {"path": location, "kind": "removed", "before": before[key]}
            else:
                yield from changes(before[key], after[key], location)
        shared_keys = before.keys() & after.keys()
        old_order = [key for key in before if key in shared_keys]
        new_order = [key for key in after if key in shared_keys]
        if old_order != new_order:
            yield {"path": path, "kind": "object_order", "before": old_order, "after": new_order}
    elif isinstance(before, list) and isinstance(after, list):
        if before == after and json.dumps(before) == json.dumps(after):
            return
        if len(before) == len(after):
            old_items = [json.dumps(item, sort_keys=True) for item in before]
            new_items = [json.dumps(item, sort_keys=True) for item in after]
            if old_items != new_items and Counter(old_items) == Counter(new_items):
                yield {"path": path, "kind": "sequence_order", "before": before, "after": after}
                return
        for index, (left, right) in enumerate(zip(before, after)):
            yield from changes(left, right, (*path, index))
        for index in range(len(after), len(before)):
            yield {"path": (*path, index), "kind": "removed", "before": before[index]}
        for index in range(len(before), len(after)):
            yield {"path": (*path, index), "kind": "added", "after": after[index]}
    elif type(before) is not type(after) or json.dumps(before) != json.dumps(after):
        record = {"path": path, "kind": "value", "before": before, "after": after}
        if type(before) in (int, float) and type(after) in (int, float):
            if all(type(value) is int or math.isfinite(value) for value in (before, after)):
                record["delta"] = after - before
        yield record


def file_diff(before, after):
    result = {
        "before_sha256": sha256(before),
        "after_sha256": sha256(after),
        "before_bytes": len(before),
        "after_bytes": len(after),
        "byte_identical": before == after,
        "changes": [],
    }
    if before != after:
        for change in changes(json.loads(before), json.loads(after)):
            change["rng"] = any(
                isinstance(part, str) and re.search(r"rng|random|seed", part, re.IGNORECASE)
                for part in change["path"]
            )
            result["changes"].append(report_value(change))
    return result


def adapt_guard(path, prefix, old, new):
    if old == new:
        return
    source = path.read_text(encoding="utf-8")
    guard = f'{prefix}"{old}"'
    if source.count(guard) != 1:
        raise ValueError(f"Expected one frozen version guard {guard!r} in {path.name}")
    path.write_text(source.replace(guard, f'{prefix}"{new}"'), encoding="utf-8", newline="\n")


def regenerate(args):
    package = Path(__file__).resolve().parents[1]
    repo = Path(git(package, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    relative = package.relative_to(repo)
    out = args.out.resolve()
    if out.is_relative_to(repo):
        raise ValueError("--out must be outside the checkout")
    if git(repo, "status", "--porcelain", "--untracked-files=normal", "--", str(relative)):
        raise ValueError("Commit package inputs before regeneration")
    source_sha = git(repo, "rev-parse", "HEAD").decode().strip()
    baseline_sha = git(repo, "rev-parse", "--verify", "--end-of-options",
                       f"{args.baseline_ref}^{{commit}}").decode().strip()
    committed_pin = git(repo, "show", f"{source_sha}:{(relative / PIN_PATH).as_posix()}")
    pin = json.loads(committed_pin)
    require_python(pin)
    out.mkdir(parents=True, exist_ok=False)
    archive = out / "source.zip"
    command = ["git", "-C", str(repo), "archive", "--format=zip", f"--output={archive}",
               source_sha, "--", str(relative)]
    subprocess.run(command, check=True)
    with ZipFile(archive) as snapshot:
        snapshot.extractall(out / "source")
    stage = out / "source" / relative
    provenance = {
        "reason": args.reason,
        "source_sha": source_sha,
        "baseline_sha": baseline_sha,
        "source_archive_sha256": sha256(archive.read_bytes()),
        "python": sys.version,
        "platform": platform.platform(),
        "requested": {"optuna": args.optuna, "optuna_mcp": args.optuna_mcp},
        "commands": [command],
        "environments": {},
        "producers": {},
    }
    provenance_path = out / "provenance.json"

    def run(command, name):
        provenance["commands"].append(command)
        write_json(provenance_path, provenance)
        with (out / f"{name}.log").open("wb") as log:
            subprocess.run(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)

    def environment(name, dependencies, torch=None):
        root = out / name
        run([sys.executable, "-I", "-m", "venv", str(root)], f"{name}-venv")
        python = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        isolated = [str(python), "-I", "-X", "utf8"]
        pip = [*isolated, "-m", "pip", "--isolated", "install"]
        if torch is not None:
            run([*pip, "--index-url", args.torch_index, "--no-deps", torch], f"{name}-torch")
        run([*pip, *dependencies], f"{name}-install")
        run([*isolated, "-m", "pip", "freeze", "--all"], f"{name}-freeze")
        provenance["environments"][name] = (out / f"{name}-freeze.log").read_text(
            encoding="utf-8"
        ).splitlines()
        return isolated

    main_pins = pin["generation"]["upstream_oracle"]
    cuda_pins = pin["generation"]["cuda_oracle"]
    optuna = f"optuna=={args.optuna}"
    torch = f"torch=={main_pins['torch']}"
    main_python = environment("venv-main", [
        optuna, f"optuna-mcp=={args.optuna_mcp}", f"numpy=={main_pins['numpy']}",
        # Match the SDK reported by the prior MCP oracle. MCP 2 removes FastMCP.
        f"mcp=={pin['generation']['mcp_oracle']['mcp']}",
        f"scipy=={main_pins['scipy']}", f"cmaes=={main_pins['cmaes']}", torch,
        "matplotlib!=3.6.0", "plotly>=4.9,<7", "pandas", "redis", "scikit-learn",
        "grpcio", "protobuf>=5.28.1",
    ], torch=torch)
    cuda_python = environment("venv-cuda-fixtures", [optuna, f"numpy=={cuda_pins['numpy']}"])
    outputs = []
    for producer, legacy, current in PRODUCERS:
        script = stage / "oracles" / producer
        original = script.read_bytes()
        prefix = {
            "generate_optuna_mcp_oracle.py": "EXPECTED_OPTUNA_VERSION = ",
            "generate_optuna50_cuda_oracle.py": "if optuna.__version__ != ",
            "generate_optuna_cuda_density_oracle.py": "assert optuna.__version__ == ",
        }.get(producer, "EXPECTED_VERSION = ")
        adapt_guard(script, prefix, pin["optuna"], args.optuna)
        if producer == "generate_optuna_mcp_oracle.py":
            adapt_guard(script, "EXPECTED_OPTUNA_MCP_VERSION = ", pin["optuna_mcp"], args.optuna_mcp)
        provenance["producers"][producer] = {
            "committed_sha256": sha256(original),
            "executed_sha256": sha256(script.read_bytes()),
        }
        legacy_output = script.with_name(legacy)
        output = script.with_name(current)
        legacy_output.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        python = cuda_python if producer in CUDA_PRODUCERS else main_python
        run([*python, str(script)], script.stem)
        if legacy_output != output and legacy_output.is_file():
            if output.exists():
                raise ValueError(f"{producer} wrote both output names")
            legacy_output.rename(output)
        outputs.append(output.relative_to(stage))

    pin.update(optuna=args.optuna, optuna_mcp=args.optuna_mcp)
    write_json(stage / PIN_PATH, pin)
    (stage / COVERAGE).unlink()
    run([*main_python, str(stage / "tools/generate_optuna_api_coverage.py")], "api-coverage")
    (stage / MANIFEST).unlink()
    run([*main_python, str(stage / "tools/generate_optuna_test_manifest.py"),
         "--test-directory", str(stage / "tests/matlab"), "--output", str(stage / MANIFEST)],
        "test-manifest")
    reports = {}
    for path in [*outputs, COVERAGE, MANIFEST]:
        before = git(repo, "show", f"{baseline_sha}:{(relative / path).as_posix()}")
        after = (stage / path).read_bytes()
        reports[path.as_posix()] = file_diff(before, after)
        destination = out / "fixtures" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(after)
    write_json(out / "diff.json", reports)
    write_json(provenance_path, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optuna", required=True)
    parser.add_argument("--optuna-mcp", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reason", default="Upstream oracle comparison")
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--torch-index", default="https://download.pytorch.org/whl/cu126")
    args = parser.parse_args()
    for version in (args.optuna, args.optuna_mcp):
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){2,3}", version):
            parser.error("upstream versions must be numeric release versions")
    regenerate(args)


if __name__ == "__main__":
    main()

"""Summarize a Flow2 job root into one result JSON and check its gates.

Usage: python summarize.py JOB_ROOT --builds build-1 build-2 --validate validate-1 --report result.json
Exits non-zero when a gate fails; the JSON is written either way.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

SKIP = {"venv", "src", "build", "__pycache__"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal(run):
    path = run / "terminal.txt"
    if not path.exists():
        return {"phase": "missing", "exit_code": None}
    values = dict(line.split("=", 1) for line in path.read_text().split() if "=" in line)
    return {"phase": values.get("phase"), "exit_code": values.get("exit_code")}


def files(run):
    return {p.relative_to(run).as_posix(): sha(p) for p in sorted(run.rglob("*"))
            if p.is_file() and not SKIP.intersection(p.relative_to(run).parts)}


def wheel_members(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return {info.filename: hashlib.sha256(archive.read(info)).hexdigest()
                for info in archive.infolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_root", type=Path)
    parser.add_argument("--builds", nargs="+", required=True)
    parser.add_argument("--validate", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.job_root.resolve()
    owner = json.loads((root / "owner.json").read_text())
    failures = []
    builds = []
    for name in args.builds:
        run = root / name
        state = terminal(run)
        record = {"run": name, "terminal": state}
        if state["phase"] == "complete":
            (wheel,) = sorted((run / "dist").glob("*.whl"))
            receipt = json.loads((run / "public-receipt.json").read_text())
            record.update(wheel=wheel.name, wheel_sha256=sha(wheel), receipt=receipt,
                          members=wheel_members(wheel))
            if receipt["source_sha"] != owner["source_sha"]:
                failures.append(f"{name}: receipt source {receipt['source_sha']} != {owner['source_sha']}")
        else:
            failures.append(f"{name}: terminal {state}")
        record["files"] = files(run)
        builds.append(record)
    complete = [b for b in builds if "wheel_sha256" in b]
    reproducibility = None
    if len(complete) >= 2:
        first = complete[0]
        differing = sorted({member for other in complete[1:]
                            for member in set(first["members"]) | set(other["members"])
                            if first["members"].get(member) != other["members"].get(member)})
        reproducibility = {
            "builds": [b["run"] for b in complete],
            "bit_identical_wheel": len({b["wheel_sha256"] for b in complete}) == 1,
            "identical_extension": len({json.dumps(b["receipt"]["python_extension_sha256s"]) for b in complete}) == 1,
            "identical_archive": len({b["receipt"]["archive_sha256"] for b in complete}) == 1,
            "differing_wheel_members": differing,
        }
    run = root / args.validate
    state = terminal(run)
    validation = {"run": args.validate, "terminal": state, "files": files(run)}
    if state["phase"] != "complete":
        failures.append(f"{args.validate}: terminal {state}")
    else:
        verified = (run / "wheel-verify.txt").read_text().split(":")[0].strip().lstrip("./")
        used = next((b for b in complete if b["wheel"] == verified), None)
        kernels = json.loads((run / "kernels-gpu.json").read_text())
        device = json.loads((run / "device.json").read_text())
        lanes = {lane: json.loads((run / f"{lane}.json").read_text()) for lane in ("primitives", "ask")}
        validation.update(
            wheel=verified, wheel_sha256=used and used["wheel_sha256"], device=device,
            gpu_statistics=kernels["statistics"], lifecycle=kernels["lifecycle"],
            stored_cases=[{k: c[k] for k in ("name", "density_max_abs_error", "best_index_one_based")}
                          for c in kernels["cases"]],
            ask_proposal_sha256={f"{c['candidates']}x{c['initial_components']}:{c['backend']}": c["proposal_sha256"]
                                 for c in lanes["ask"]["cases"]},
            lane_ok={lane: report["ok"] for lane, report in lanes.items()},
        )
        if used is None:
            failures.append(f"validated wheel {verified} is not a completed build")
        if not kernels["ok"] or kernels["backend"] != "gpu":
            failures.append("GPU kernel verification did not pass")
        if kernels["statistics"].get("cuda_device", -1) < 0 or kernels["statistics"]["density_launches"] <= 0:
            failures.append("no GPU execution recorded (CPU fallback is not allowed)")
        if not all(validation["lane_ok"].values()):
            failures.append(f"oracle lanes failed: {validation['lane_ok']}")
    result = {
        "schema": "radia.optuna.flow2-reproducible-route.v1",
        "owner": owner, "builds": [{k: v for k, v in b.items() if k != "members"} for b in builds],
        "reproducibility": reproducibility, "validation": validation,
        "inputs_sha256": sha(root / "inputs.sha256"), "ok": not failures, "failures": failures,
        "scope": "Build and installed-wheel GPU correctness on Flow2; elapsed times are not performance measurements",
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()

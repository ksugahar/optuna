"""Guards: the Flow2 summary attributes a validation to the exact wheel and job.

Assumes: tools/flow2/summarize.py reads the job-root layout written by
build.pbs and validate.pbs. Synthetic roots only; no PBS or GPU is used.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/flow2/summarize.py"
spec = importlib.util.spec_from_file_location("flow2_summarize", SCRIPT)
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)

SOURCE = "a" * 40
WHEEL = "radia_optuna-5.1.0.2-cp312-cp312-linux_aarch64.whl"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def finish(run, job, exit_code="0"):
    (run / "terminal.txt").write_text(f"phase=complete\nexit_code={exit_code}\n")
    write_json(run / "job.json", {"run": run.name, "pbs_jobid": job, "node": "n1"})


def make_build(root, name, payload, job):
    run = root / name
    (run / "dist").mkdir(parents=True)
    wheel = run / "dist" / WHEEL  # same filename in every build, bytes may differ
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("radia_optuna/_optuna_kernels.so", payload)
    (run / "wheel.sha256").write_text(f"{digest(wheel)}  ./{WHEEL}\n")
    write_json(run / "public-receipt.json", {
        "source_sha": SOURCE, "archive_sha256": "b" * 64,
        "python_extension_sha256s": {"_optuna_kernels.so": hashlib.sha256(payload).hexdigest()}})
    finish(run, job)


def make_validation(root, build, cuda_device=0):
    run = root / "validate-1"
    run.mkdir()
    wheel = root / build / "dist" / WHEEL
    write_json(run / "validated-wheel.json", {
        "build": build, "wheel": WHEEL, "sha256": digest(wheel),
        "receipt_sha256": digest(root / build / "public-receipt.json"), "source_sha": SOURCE})
    write_json(run / "kernels-gpu.json", {
        "ok": True, "backend": "gpu", "lifecycle": {}, "cases": [],
        "statistics": {"cuda_device": cuda_device, "density_launches": 6}})
    write_json(run / "device.json", {"gpus": [{"name": "GB200", "compute_cap": "10.0"}]})
    write_json(run / "primitives.json", {"ok": True, "cases": []})
    write_json(run / "ask.json", {"ok": True, "cases": [
        {"candidates": 24, "initial_components": 32, "backend": "gpu", "proposal_sha256": "c" * 64}]})
    finish(run, "300.jobm")


@pytest.fixture
def root(tmp_path):
    write_json(tmp_path / "owner.json", {"source_sha": SOURCE})
    (tmp_path / "inputs.sha256").write_text("x\n")
    make_build(tmp_path, "build-1", b"first", "100.jobm")
    make_build(tmp_path, "build-2", b"second", "200.jobm")
    return tmp_path


def run(root):
    return summarize.summarize(root, ["build-1", "build-2"], "validate-1")


def test_validation_of_build_2_is_attributed_to_build_2(root):
    make_validation(root, "build-2")
    result = run(root)
    assert result["failures"] == [], result["failures"]
    assert result["validation"]["validated_wheel"]["build"] == "build-2"
    assert result["validation"]["validated_wheel"]["sha256"] == digest(root / "build-2/dist" / WHEEL)
    assert result["pbs_jobids"] == {"build-1": "100.jobm", "build-2": "200.jobm", "validate-1": "300.jobm"}
    assert result["reproducibility"]["bit_identical_wheel"] is False
    assert result["reproducibility"]["differing_wheel_members"] == ["radia_optuna/_optuna_kernels.so"]


def test_wheel_modified_after_validation_is_rejected(root):
    make_validation(root, "build-2")
    with zipfile.ZipFile(root / "build-2/dist" / WHEEL, "a") as archive:
        archive.writestr("extra", b"tampered")
    failures = run(root)["failures"]
    assert any("wheel.sha256" in f for f in failures), failures
    assert any("validated wheel identity" in f for f in failures), failures


def test_identity_naming_the_other_build_is_rejected(root):
    make_validation(root, "build-2")
    path = root / "validate-1/validated-wheel.json"
    identity = json.loads(path.read_text())
    write_json(path, dict(identity, build="build-1"))  # same filename, build-2 bytes
    failures = run(root)["failures"]
    assert any("validated wheel identity" in f for f in failures), failures


def test_complete_marker_with_nonzero_exit_fails(root):
    make_validation(root, "build-2")
    finish(root / "build-1", "100.jobm", exit_code="1")
    failures = run(root)["failures"]
    assert any(f.startswith("build-1: terminal") for f in failures), failures


def test_missing_gpu_execution_fails(root):
    make_validation(root, "build-2", cuda_device=-1)
    failures = run(root)["failures"]
    assert any("CPU fallback" in f for f in failures), failures

#!/bin/bash
# Type I step of type1.pbs: install one SHA-256-verified CUDA-free wheel into a
# clean venv, check it against the pinned upstream Optuna oracle, then time both
# oracle lanes (upstream NumPy and native CPU) at each pinned thread count.
# Usage (inside the job): JOB_ROOT=<root> RUN=<validate-N> BUILD=<build-N> bash validate_cpu.sh
set -Eeuo pipefail
umask 077
: "${JOB_ROOT:?}" "${RUN:?}" "${BUILD:?}"
sweep=${THREADS_FLOW2:-1 16 48}
run=$JOB_ROOT/$RUN
mkdir "$run"
cd "$run"
exec >run.log 2>&1
phase=allocation
trap 'rc=$?; printf "phase=%s\nexit_code=%s\n" "$phase" "$rc" > terminal.txt' EXIT
test "$(uname -m)" = x86_64
python3.12 -c 'import json,os,socket,sys; json.dump({"run":sys.argv[1],"pbs_jobid":os.environ["PBS_JOBID"],"node":socket.gethostname()},open("job.json","w"),indent=2)' "$RUN"
(cd "$JOB_ROOT" && sha256sum --quiet -c inputs.sha256)
grep -qx 'phase=complete' "$JOB_ROOT/$BUILD/terminal.txt"
grep -qx 'exit_code=0' "$JOB_ROOT/$BUILD/terminal.txt"
(cd "$JOB_ROOT/$BUILD/dist" && sha256sum -c ../wheel.sha256) > wheel-verify.txt
wheels=("$JOB_ROOT/$BUILD"/dist/*.whl)
test ${#wheels[@]} -eq 1
wheel=${wheels[0]}
# Identity of the exact file installed below: build, filename, observed hash, source.
python3.12 - "$BUILD" "$wheel" "$JOB_ROOT/$BUILD/public-receipt.json" > validated-wheel.json <<'PY'
import hashlib, json, pathlib, sys
build, wheel, receipt = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
json.dump({"build": build, "wheel": wheel.name,
           "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
           "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
           "source_sha": json.loads(receipt.read_text())["source_sha"]}, sys.stdout, indent=2)
PY
python3.12 - "$JOB_ROOT/owner.json" <<'PY'
import datetime, json, sys
deadline = datetime.datetime.fromisoformat(json.load(open(sys.argv[1]))["deadline_utc"])
assert datetime.datetime.now(datetime.timezone.utc) < deadline, "past the owner.json deadline"
PY
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
phase=environment
python3.12 -m venv venv   # clean environment: only hash-pinned wheels below
export PATH="$run/venv/bin:$PATH"
src=$JOB_ROOT/source/radia-optuna
python -m pip install -q --no-index --require-hashes --find-links "$JOB_ROOT/wheelhouse-validate" \
  -r "$src/tools/flow2/requirements-validate.txt"
python -m pip install -q --no-index --no-deps "$wheel"
python -m pip freeze > freeze.txt
receipt=$(python -I -c 'import pathlib, radia_optuna; print(pathlib.Path(radia_optuna.__file__).with_name("native_build.json"))')
case $receipt in "$JOB_ROOT/source"/*) echo "radia_optuna resolved inside the checkout" >&2; exit 1;; esac
cmp "$receipt" "$JOB_ROOT/$BUILD/public-receipt.json"   # installed receipt is the build's
phase=topology
# shellcheck disable=SC2086
python "$src/tools/flow2/cpu_topology.py" --threads $sweep --report topology.json
cpus() { python -c 'import json,sys; print(",".join(map(str,json.load(open("topology.json"))["selections"][sys.argv[1]]["cpus"])))' "$1"; }
phase=kernels-cpu
OMP_NUM_THREADS=1 taskset -c "$(cpus "${sweep%% *}")" \
  python "$src/validation_test/validate_kernels.py" --backend cpu --installed-wheel \
  --build-receipt "$receipt" --report kernels-cpu.json
for n in $sweep; do
  for lane in primitives ask; do
    phase=timing-$lane-t$n
    OMP_NUM_THREADS=$n taskset -c "$(cpus "$n")" \
      python "$src/validation_test/benchmark_flow2_ask.py" "$lane" --installed-wheel \
      --build-receipt "$receipt" --backends numpy cpu --report "$lane-t$n.json"
  done
done
phase=complete

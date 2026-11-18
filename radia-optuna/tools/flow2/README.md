Flow2 Linux Python GPU route
============================

This route builds the radia-optuna Linux Python/pybind11 GPU wheel on Flow2
(Grace + GB200, aarch64) from one exact commit. It then installs that same
SHA-256-verified wheel into a clean environment and checks it on the GPU
against the pinned upstream Optuna oracle. Flow2 is outside GitHub Actions and
has no runner. The wheel is a job-local qualification artifact and is never
published.

Fixed inputs
------------

| Item | Value |
|---|---|
| Source | one full 40-hex commit; `build_linux_python.py` rejects tracked changes |
| Python | system `python3.12` (3.12, GCC 11.5 build) |
| Build wheels | `requirements-build.txt`, hash-pinned (pybind11 3.0.2, CMake, Ninja, setuptools, wheel) |
| Oracle runtime | `requirements-validate.txt`, hash-pinned; Optuna equals `src/radia_optuna/upstream.json` |
| CUDA | facility CUDA 12.8 (`/home/center/opt/aarch64/cores/cuda/12.8`), `nvcc` absolute path |
| Host compiler | `/usr/bin/g++` (GCC 11.5) |
| Architecture | `sm_100` (`CUDA_ARCH_FLOW2=100`) |
| GPU check | exactly one visible GPU, name contains `GB200`, compute capability `10.0` |
| Threads | `OMP_THREADS_FLOW2` (default 8), pinned to the first N CPUs of the allocation |

Override the toolchain only with `CUDA_HOME_FLOW2`, `CXX_FLOW2`,
`CUDA_ARCH_FLOW2`, `EXPECT_GPU_NAME` and `EXPECT_GPU_CC`. There is no silent
default substitution: a missing tool or a different GPU fails the job.

Steps
-----

1. On the login node, stage a new job root. The root must not exist yet.
   `SOURCE` is a git URL or a bundle that contains the commit.

   ```sh
   tools/flow2/stage.sh SOURCE SHA "$HOME/<jobs>/<root>" "<owner>" "<request>" 6
   ```

   This clones the exact commit, downloads the hash-pinned wheels for
   aarch64/CPython 3.12, and writes `owner.json` (owner, request, source SHA,
   deadline) and `inputs.sha256`. Compute jobs need no network access.

2. Submit two independent builds and one validation on `t2-share`, which
   allocates a GB200. `t2-dev` gave a MIG slice of the GPU, runs one job per
   user and allows at most one hour, so use it only for a quick smoke test.
   The builds may run in parallel; the validation waits for both:

   ```sh
   t=$ROOT/source/radia-optuna/tools/flow2
   b1=$(qsub -q t2-share -v JOB_ROOT=$ROOT,RUN=build-1 "$t/build.pbs")
   b2=$(qsub -q t2-share -v JOB_ROOT=$ROOT,RUN=build-2 "$t/build.pbs")
   qsub -q t2-share -W depend=afterok:$b1:$b2 -v JOB_ROOT=$ROOT,RUN=validate-1,BUILD=build-1 "$t/validate.pbs"
   ```

   Each build clones its own tree, sets `SOURCE_DATE_EPOCH` to the commit
   time, builds the extension with `build_linux_python.py`, and runs
   `pip wheel --no-index --no-build-isolation`. The validation job creates a
   clean venv, installs only the hash-pinned oracle runtime and the verified
   wheel, and checks that `radia_optuna` resolves outside the checkout. It then
   runs `device_identity.py`, `validate_kernels.py --backend gpu
   --installed-wheel`, and both `benchmark_flow2_ask.py` lanes with
   `--installed-wheel`.

3. Summarize, then recover:

   ```sh
   python3.12 $t/summarize.py $ROOT --builds build-1 build-2 --validate validate-1 --report $ROOT/result.json
   ```

   Copy `result.json` and each run directory (without `venv/`, `src/` and
   `build/`) to private evidence storage. Verify the SHA-256 values after the
   transfer, and only then delete the job-owned root and confirm it is gone.

Gates and failure handling
--------------------------

Every job writes `job.json` (run, PBS job ID, node) and `terminal.txt` with the
last phase and exit code. A run counts as finished only with `phase=complete`
and `exit_code=0`. The validation job writes `validated-wheel.json`: the build
name, wheel filename, SHA-256 of the exact file it installed, the receipt hash
and the source SHA. `result.json` lists all PBS job IDs. `summarize.py` exits
non-zero, still writing `result.json`, when any of these hold:

- a run lacks a job ID, or is not `complete` with exit code 0;
- a build does not have exactly one wheel, its bytes differ from its own
  `wheel.sha256`, or its receipt names another source commit;
- the validated wheel identity does not match the selected build in every field
  (builds normally share a filename, so the filename alone identifies nothing);
- GPU kernel verification failed, or no GPU launch or device was recorded
  (CPU fallback is not allowed). Only this GPU check must prove GPU execution;
  the NumPy and native CPU backends in the oracle lanes are parity references;
- an oracle lane failed: stored density/selection and model cases, or a
  seeded `Study.ask()` proposal SHA-256 or winner list differs between upstream
  NumPy, native CPU and GPU.

Do not edit a job root after a failure. Fix the cause, then stage a new root.
Oracle fixtures and tolerances are unchanged by this route.

Reproducibility
---------------

`result.json` reports `bit_identical_wheel`, `identical_extension`,
`identical_archive` and the differing wheel members across the two builds. If
the wheels are identical, the route gives bit-identical independent rebuilds
on this toolchain. Otherwise it gives provenance reproducibility: the same
commit, pinned inputs and recipe, with the differing members listed.

Observed on CUDA 12.8 / GCC 11.5: two builds of one commit produced extensions
of equal size that differ in content. nvcc names its intermediate file
`tmpxft_<process id>_...` and embeds that name in the host object (for example
in the static initializer symbol), so the extension, the archive, the wheel
receipt and `RECORD` differ between builds. The route therefore gives
provenance reproducibility, not bit-identical rebuilds.

Elapsed times recorded by the oracle lanes describe that job only. They are
not performance measurements.

Flow2 Linux Python routes
=========================

Flow2 is the primary Linux environment for radia-optuna. Two lanes build the
Linux Python/pybind11 wheel from one exact commit, install that same
SHA-256-verified wheel into a clean environment and check it against the pinned
upstream Optuna oracle:

- Type II (Grace + GB200, aarch64): GPU profile, built on `t2-dev`, validated
  on a full GB200 on `t2-share`.
- Type I (x86_64): CUDA-free CPU profile on `t1-single`, reference agreement
  and timings at 1, 16 and 48 threads (see "Type I lane" below).

Flow2 is outside GitHub Actions and has no runner. Linux wheels are never
uploaded to PyPI. Qualified wheels reach Flow2 users only through the lab
distribution described under "Distribution to Flow2 users" below. This route is
the reference that other packages follow for Flow2 distribution.

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
   tools/flow2/stage.sh SOURCE SHA "$HOME/<jobs>/<root>" "<owner>" "<request>" 6 aarch64
   ```

   This clones the exact commit, downloads the hash-pinned wheels for
   CPython 3.12 on the named machine (`aarch64` for Type II, `x86_64` for
   Type I), and writes `owner.json` (owner, request, source SHA, machine,
   deadline) and `inputs.sha256`. Compute jobs need no network access. Each
   requirement pins one hash per platform wheel.

2. Build on `t2-dev` and validate on a GB200 on `t2-share`. `t2-dev` gives
   4 CPUs and a MIG slice of the GPU, allows one hour and runs one job per
   user, so the two builds are chained. A build took under a minute there.
   If a build does not fit `t2-dev`, report it; do not move it to another
   queue silently. The GB200 validation waits for both builds:

   ```sh
   t=$ROOT/source/radia-optuna/tools/flow2
   b1=$(qsub -q t2-dev -l gpu=1 -v JOB_ROOT=$ROOT,RUN=build-1 "$t/build.pbs")
   b2=$(qsub -q t2-dev -l gpu=1 -W depend=afterok:$b1 -v JOB_ROOT=$ROOT,RUN=build-2 "$t/build.pbs")
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

Type I lane
-----------

Type I has no CUDA, and the kernels need none: `build.pbs` with `PROFILE=cpu`
compiles `native/optuna_host.cpp` (the shared kernel source) with the system
`/usr/bin/g++` (GCC 11.5, the same host compiler as the Type II nvcc build).
The CPU equations do not call BLAS, so no MKL or other library enters the
extension. The `gpu` backend of this build raises.

`t1-single` allocates a whole node (it requires `-l node=1`; 384 CPUs), so one job runs every
step, each in its own run directory with the same PBS job ID:

```sh
tools/flow2/stage.sh SOURCE SHA "$HOME/<jobs>/<root1>" "<owner>" "<request>" 6 x86_64
qsub -q t1-single -l node=1 -v JOB_ROOT=$ROOT1 "$ROOT1/source/radia-optuna/tools/flow2/type1.pbs"
python3.12 $t/summarize.py $ROOT1 --builds build-1 build-2 --validate validate-1 \
  --profile cpu --threads 1 16 48 --report $ROOT1/result.json
```

`type1.pbs` runs build-1 and build-2, then `validate_cpu.sh` on build-1's
verified wheel. That step runs `validate_kernels.py --backend cpu` (stored
oracle cases; it also checks that the gpu backend is rejected), records
`topology.json` (`lscpu -J`, `numactl --hardware` when present, and the chosen
CPUs), then runs both oracle lanes with `--backends numpy cpu` at 1, 16 and 48
threads. Each thread count is pinned with `taskset` to one hardware thread per
physical core, filled socket by socket and NUMA node by NUMA node, with
`OMP_PROC_BIND=close` and `OMP_PLACES=cores`. Each lane report records
`OMP_NUM_THREADS` and three CPU masks:
- the process mask at launch, read before any OpenMP runtime loads;
- the main-thread mask after the native module loads;
- the mask of every live thread at the end of the lane.
With `OMP_PROC_BIND` set, libgomp binds the initial thread to its first place
when it initializes, so the main-thread mask shows one CPU (observed on Flow2:
`[0]` at 16 and 48 threads). The launch mask must equal the selection, and
every thread mask must lie inside it. Every native CPU case also records the absolute
team size of its last host density (`native_team`), because the in-process
maximum cannot be reduced to a per-case delta. The 4096x512 case always enters
the parallel density region, so its team must equal the requested thread
count; smaller cases report the team they actually used.

The CPU summary fails when the build is not the host-only route or accepts the
gpu backend, when a lane launched off its core selection or thread count or a
thread mask left the selection, when the
4096x512 native team differs from the thread count, when an
oracle lane failed, or when a seeded `Study.ask()` proposal differs between
upstream NumPy and native CPU or across thread counts. `result.json` lists the
median time per operation for every thread count, lane, size and backend. These
timings are diagnostic: they never gate.

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
of equal size that differ in content; the archive, the wheel receipt and
`RECORD` differ as a consequence. One identified difference is that nvcc names
its intermediate file `tmpxft_<process id>_...` and embeds that name in the
host object (for example in the static initializer symbol). Other differing
bytes were not isolated, so this name is not claimed to explain them all. The
GPU route therefore gives provenance reproducibility, not bit-identical
rebuilds. The CUDA-free Type I build reports its own comparison.

Elapsed times recorded by the oracle lanes describe that job only. They are
not performance measurements.

Distribution to Flow2 users
---------------------------

The lab distribution is the primary route for Flow2 users, for example
students running their own jobs under their own accounts. Users install from
the lab's shared distribution folder, never from a job root or a build tree.
The folder location is lab configuration and is not recorded in this
repository. Attaching the same verified files to a GitHub release is optional.

The rules in items 1 to 5 are the general Flow2 reference that other packages
follow. "Optuna settings" at the end lists what is specific to radia-optuna.

1. Eligibility. Only a release whose exact source commit is merged is
   distributed. Every qualification lane of the package must be complete at
   that exact commit. Evidence from an earlier commit, including a pre-merge
   candidate, does not qualify a later merged commit, even with an equal
   tree; the lanes run again at the merged commit. Completion means:
   - every PBS job ID is recorded, and every run is `complete` with exit code 0;
   - every `result.json` summary passes;
   - the wheels, logs, result JSON and artifacts are recovered to private
     evidence storage, with SHA-256 matching at both ends.
   A missing or mismatched hash, a failed gate or an unrecorded job means no
   distribution. Placing files also needs the release owner's explicit approval.
2. Content. One folder per package and source commit, holding only:
   - each project wheel, as the exact file its validation job installed
     (`validated-wheel.json`); wheels are never rebuilt for distribution;
   - `wheelhouse/`: the hash-pinned runtime dependency wheels for every
     distributed architecture. These are the files the qualification jobs
     installed from their staged wheelhouses, checked against the pinned
     requirement files;
   - the pinned requirement file of the runtime;
   - PBS examples for user jobs;
   - a short README with the source SHA, PBS job IDs, Python version and the
     profile of each wheel;
   - `SHA256SUMS`, covering every other file in the folder (payload files,
     `wheelhouse/` included) and not itself.
   The distribution folder keeps only the latest verified release. A
   superseded release is removed after its replacement is verified.
3. Installation. A PBS example creates a job-local venv and installs offline
   from the folder only:

   ```sh
   (cd "$DIST" && sha256sum -c SHA256SUMS)
   python3.12 -m venv venv
   venv/bin/python -m pip install --no-index --require-hashes \
     --find-links "$DIST/wheelhouse" -r "$DIST/requirements-validate.txt"
   venv/bin/python -m pip install --no-index --no-deps "$DIST/<project wheel for this node>"
   ```

   The runtime dependencies resolve first from the hash-pinned wheelhouse.
   The verified project wheel is then installed by its exact path with
   `--no-deps`; its hash was checked by `SHA256SUMS`.
4. Acceptance before announcement. From a separate user account, without
   Administrator paths, fetch the folder and run each PBS example to
   completion on each node type: manifest check, fresh venv, installation,
   and one run of the package. Record the job IDs and outputs with the release
   evidence. Until then the release is not announced.
5. Version control. Distribution files are not committed to a source tree.
   When the folder sits inside a checkout, it must be a path that Git ignores.

Optuna settings (specific to radia-optuna):

- Lanes: Type II (aarch64, GPU profile): two builds on `t2-dev`, GB200
  validation on `t2-share`. Type I (x86_64, CUDA-free CPU profile): one
  `t1-single` job (`-l node=1`) with two builds, reference agreement and
  1/16/48-thread timings. Distributed wheels: `linux_aarch64` from the Type II
  `validate-1` build and `linux_x86_64` from the Type I `validate-1` build.
- Runtime: CPython 3.12. `requirements-validate.txt` pins the oracle runtime
  (pinned upstream Optuna and its dependencies) with one hash per platform
  wheel. `wheelhouse/` holds the union of the two staged `wheelhouse-validate`
  directories.
- Qualification: the gates in this README. Timings are diagnostic and never
  gate.
- PyPI: Linux wheels are never uploaded. PyPI publication accepts only
  `win_amd64` wheels.

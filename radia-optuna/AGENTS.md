# radia-optuna execution policy

Develop and verify the shared CUDA kernels through the Python/pybind11 binding
first. Compile the common .cu source once with MATLAB-bundled nvcc and place
its object in a CMake static library. Link that same verified artifact into
pybind11 and the thin plain MSVC MEX adapter using matlab_add_mex. The mexcuda
build route is retired; do not retain it as a fallback.

TPE supports explicit CPU (OpenMP host kernels) and GPU (CUDA kernels)
selection. The MEX adapter must not include or link NGSolve TaskManager.
Implicit GPU-to-CPU fallback is prohibited. Verify CPU execution creates no
CUDA context and verify same-kernel Python/MEX parity before qualification.

Preserve pinned upstream Optuna equations, ordered RNG consumption, first-maximum ties
and failure-state restoration on both routes. Validate both modes. Treat
execution selection as process-local configuration, separate from Study
storage; parallel workers require their own explicit selection.

Development order (2026-11-17): implement and check the CPU base (pybind11
and MATLAB) on the build host, where CPU CI runs; then develop the
Python/pybind11 GPU route on Flow2; then the MATLAB GPU route on A100.

To move to a newer upstream Optuna release, follow
`.claude/skills/optuna-upstream-follow/SKILL.md`.

---
name: optuna-upstream-follow
description: Move radia-optuna from its pinned upstream Optuna release to a newer upstream release. Regenerate the upstream-generated oracles, adopt the new design in place, and requalify CPU/GPU and installed-wheel behavior. Use when a new Optuna release appears, or when asked to bump, follow or track upstream Optuna.
---

# Follow an upstream Optuna release

radia-optuna treats one pinned Optuna release as the only oracle for shared
behavior: seed, options, parameter and search-space order, history, named
constraints, values, states, warnings, defaults, extension points and random
consumption. Following upstream means moving that pin to a newer **release**.
Do not track upstream `main`. Handwritten MATLAB output is never the expected
value.

## 0. Decide whether to move

1. Run `python tools/upstream_watch.py` (or read the latest run of the
   `radia-optuna upstream watch` workflow). It names the newer stable tag,
   both commit SHAs and the changed paths by component. Its review points are
   inferred from paths only. Then read the upstream release notes and the diff
   between the current pin and the new release. Note especially any change to samplers, the RNG, TPE, the
   search space, storage or the public API.
2. Check that `optuna-mcp` has a release compatible with the new Optuna
   version. The MCP oracle records the Optuna and optuna-mcp versions together,
   so the two pins move together.
3. Write a short plan for the user that says what changes upstream, which
   oracles are affected, and which MATLAB APIs are added or removed. The move
   starts only after the user agrees.

## 1. Prepare

- Work on a branch from the current `main`, in a clean checkout.
- Create an isolated venv with the exact new versions: `optuna==<new>`, the
  matching `optuna-mcp`, and the NumPy version you record. Never use a shared
  runtime.
- Record the interpreter, all package versions and the source SHA for every
  generation run.

## 2. Make the pin a single source (before the first move only)

The pin is written out in many places, and the oracle files, producers and
the API coverage file carry the version in their names (`optuna50_*`). Before
the first move, rename them to version-neutral names in a separate reviewed
commit that changes no expected values:

- Fixtures and producers stay byte-identical wherever possible. Change only
  file names and the live locators that point at them.
- Update every live locator in the same commit. That includes the source
  identity contract (`oracles/README.md` and `tests/test_cuda_oracle_provenance.py`),
  the distribution contract, `src/radia_optuna/evidence_manifest.json`,
  `setup.py`, `verify_wheel.py` and its tests, `tools/build_native.py`,
  `tools/generate_optuna_api_coverage.py`, `tools/generate_optuna_test_manifest.py`,
  the test manifest, the MATLAB consumers in `tests/matlab/` and
  `validation_test/`, and `matlab/` metadata. Use `git grep optuna50_` to find
  them.
- Keep the CUDA producer/fixture pair (`oracles/generate_optuna50_cuda_oracle.py`
  and `oracles/optuna50_cuda_oracle.json`) at its original paths. The
  producer writes the fixture by a hard-coded file name, and its
  LF-normalized bytes are hashed into that fixture. Renaming it now would
  either change those bytes or leave it writing to an obsolete name. Rename
  the pair in the approved regeneration (step 3), and update its embedded
  hash and all its locators together then.
- Recorded provenance is immutable. Do not rewrite historical command paths,
  recorded versions or hashes in fixtures, evidence or review records.
- Rebuild the installed evidence ledger, build the wheel and run
  `verify_wheel.py` on it. The README and source contract alone do not prove
  the rename.

Make one value the source of the pin, and have the other active pins read it
or be checked against it by a contract.

## 3. Regenerate the oracles

Follow the regeneration rules in `oracles/README.md`. Regeneration is an
explicit, reviewed change that needs approval for expected-value changes. It
is never a test setup step. Preserve the old evidence in Git.

Use the scratch-only command in `oracles/README.md` to run all five producers
and the two derived generators in isolated environments. Review
`provenance.json` and `diff.json` before adopting any output. If the producers
have been updated in the migration commit, use `--baseline-ref` to compare
against the prior reviewed fixture tree. Confirm the CUDA selection/model
fixture's embedded fingerprint against the exact producer in the proposal.

For every fixture, record the reason, the source identity, the full dependency
and command provenance, the raw byte and hash diff, the case ordering and the
numerical differences. Update the reviewed identity hashes in
`oracles/README.md` and in the source contracts.

## 4. Adopt the new design in place

There is no backward compatibility. When upstream removes or renames an API,
an option or a behavior, remove or rename it in the MATLAB package too. Delete
the old fixtures and pins instead of keeping them beside the new ones. Do not
add compatibility shims, aliases or flags. New upstream behavior that the
package shares is implemented to match the new oracle.

## 5. Set the release version

Set the package version and update the CHANGELOG before qualification. An
upstream migration uses the pinned Optuna release version. A Radia-only
release on that pin increments the fourth release segment (`major.minor.patch.revision`).
The release workflow binds the `pyproject.toml` version, the
`radia-optuna-v<version>` tag, the successful main CI run and the wheel's
native receipt to one source SHA. A version bump after qualification would
change that SHA and the wheel, so the qualified wheel could not be published.

After the bump, `git grep` for the old version must find no active pins or
locators. Recorded provenance and the CHANGELOG may still name it.

## 6. Development qualification on the PR candidate

Run these on the PR head SHA from step 5. If the candidate changes, run them
again on the new head.

1. Source contracts: `python -m pytest tests -q` with the new pin.
2. MATLAB oracle tests against the new fixtures:
   `tests/matlab/test_optuna_upstream_oracle.m` and
   `tests/matlab/test_optuna_mcp_oracle.m`.
3. Native development builds, in this order (user 2026-11-17):
   - On the build host: compile the CUDA-enabled Windows bindings once.
     The Windows pybind11 module and the plain MEX link the same Windows nvcc
     object/archive, and the CPU Python proof comes before the MEX link.
     Exercise both bindings in CPU mode only: no CUDA context, bitwise
     pybind11/MEX parity and CPU TPE parity. The CPU CI jobs run here.
   - On Flow2: build the Linux Python/pybind11 route from the same common CUDA
     source with `tools/build_linux_python.py` (without MATLAB or Simulink),
     then run `validation_test/validate_kernels.py` for the CPU and GPU
     backends. Flow2 contributes Python GPU evidence only; no Linux object
     goes to A100.
   - On A100: run the Windows bindings from the build host in GPU mode:
     binding parity, CUDA validation, GPU TPE, GPU Simulink and Engine.

   These builds find problems early. They do not qualify the release wheel.
   Record the SHA they ran on.
4. Confirm that CPU and GPU agree within the existing tolerances.

If a result differs from the new oracle, stop. Find the cause in the upstream
diff or in this package. Never widen a tolerance or edit an expected value to
make a test pass. Changing a tolerance or another acceptance criterion needs
the user's approval and two independent checks.

## 7. Review, merge and official qualification

1. An independent reviewer checks the exact PR head SHA before merge. The
   review covers oracle independence: the expected values come from
   upstream, not from this package's own output.
2. Merge the PR. Check that the tree of the resulting `main` commit equals
   the reviewed tree (`git rev-parse <main>^{tree}` against the reviewed
   head's tree). If it differs, review the difference before going on.
3. Dispatch the native CI (`.github/workflows/radia-optuna.yml`, which runs
   only by `workflow_dispatch` on `main`) on that main SHA. Its build job
   produces the formal Windows wheel. The GPU acceptance, Engine and HDF5
   storage jobs install that exact wheel and check its SHA-256. Only this
   wheel is a release candidate. When the main SHA differs from the SHA of
   the development builds in step 6, record both.
4. Tag exactly the main SHA whose native CI passed as
   `radia-optuna-v<version>`, and release it through
   `release-radia-optuna.yml`. The release verifier requires a successful
   main run for the tag SHA. Do not add commits between that run and the tag.
5. After the release, update the places outside this repository that name the
   pin, for example the Radia repository policy and the radia-mcp Optuna
   knowledge.

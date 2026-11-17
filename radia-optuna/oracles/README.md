# CUDA oracle source evidence

Run oracle producers in a separate regeneration directory. The regeneration
tool also maps historical output names when reproducing older commits.

Stored oracle inputs originate in upstream Optuna.
The CUDA selection/model and density fixtures below were regenerated with
Optuna 5.1.0. Python 3.12.10 and NumPy 2.5.3 were used. The structured
diff against the prior reviewed inputs found only upstream metadata and
the selection/model producer fingerprint changes. Case inputs, outputs,
ordering, and recorded RNG consumption matched.
The six JSON inputs (including MATLAB classification metadata) are wheel
payloads under `radia_optuna/oracles`. Producers stay source-only and are
never imported or executed by wheel construction or verification.

The files were originally relocated from `tests/matlab/fixtures/`
at source `f669767b7d1f0de36e7f590a97717adc0038dfd7`. The regenerated fixture records
Optuna 5.1.0 and NumPy 2.5.3. Its six selection cases and fourteen model cases,
including their order, values and random consumption, are unchanged.

| Input | SHA-256 of Git blob bytes |
| --- | --- |
| `generate_optuna50_cuda_oracle.py` | `b5e49a9bea5253c8e1a7b55222a3a2a66a3915da475797c5d1248f8ecb0dd7dc` |
| `optuna50_cuda_oracle.json` | `73375a7e511cfccaa4916b4304c433c9f53e3d881df0b10c9dc5a3c29d0fa4d8` |

The producer hash embedded in the fixture uses LF-normalized bytes, matching
its existing generation procedure. Git pins the fixture itself to LF. The
source contract preserves these approved identities and the package's upstream
pin; it does not regenerate numerical expectations or certify GPU execution.

Regeneration is an explicit, reviewed change, never a test setup step. Before
running the producer, obtain approval for expected-value changes, use an
isolated environment matching the recorded upstream versions, and preserve the
old evidence in Git. Record the reason, source identity, full dependency and
command provenance, raw byte/hash diff, case ordering and numerical differences.
Review the proposed fixture and contract hash updates independently before
accepting them. Keep the source, environment, and diff for each generation.

## Density evidence

The additional density pair was relocated byte-for-byte from source
`53b4f2cfc01931398acf6b978b9180eef1752f5d`, without importing or executing its
producer (which writes the fixture at module import).

| Input | SHA-256 of Git blob bytes |
| --- | --- |
| `generate_optuna_cuda_density_oracle.py` | `3cf68642bd20dd9d1eafe32f659525689cdc26843f7fbbc83a1bb2f1e5e49b36` |
| `optuna_cuda_density_oracle.json` | `167f8bef1d1088ab5c55c6bb2845c97e6084a496e665d672e9e877b20697545b` |

This stored schema records Optuna 5.1.0 and cases, but no NumPy version or
embedded producer hash. The companion producer is identified by its Git bytes;
no missing generation metadata is inferred from the other oracle or invented.
The same explicit review and regeneration policy above applies.

## Official MCP oracle

`generate_optuna_mcp_oracle.py` and `optuna_mcp_oracle.json` are
inputs moved unchanged from commit
`abad8cc4c79faea5e068f3c78ab01b729c407d5c`. The stored fixture records
Optuna 5.1.0, optuna-mcp 0.2.0 and stdio transport.
Regeneration with the prior MCP SDK (1.27.0) changed only the Optuna version. It has no embedded
producer fingerprint; the source identity contract records the reviewed
producer and fixture bytes without inventing generation metadata. The
producer must not be executed during relocation or ordinary source checks.

| Input | Git blob SHA256 |
| --- | --- |
| `generate_optuna_mcp_oracle.py` | `b1ecc55ace63671b153fceaa5b90f0a3cbf3b25f855ce9f5de758294ab399ba6` |
| `optuna_mcp_oracle.json` | `894dff133d12e2f2e13058b17bfa4dd5e81aadf8c36668042e0703b0d7f3cd0e` |

## Public API inventory

The prior Optuna 5.0.0 public inventory and its producer were moved unchanged
from commit `dfe22b6ceb6dc0a55b902c4a80180176dfcfa814`. The stored inventory has no embedded
producer hash. The source contract fixes the reviewed identities below;
relocation did not run the producer. The 5.1.0 regeneration changed only the
upstream version and the NSGA-III reference-point input type in the inventory.

| Input | Git blob SHA256 |
| --- | --- |
| `generate_optuna_api_inventory.py` | `bcc580d912b0e5f724d8e9ece58d48674f267a208a792ffe535fabdce81ece12` |
| `optuna_public_api.json` | `f4b8bd46dd72bcafc8bd79651f368d90e7d77eeb318ad765f30b2363137bf678` |

## Main upstream behavior oracle

The original producer and fixture were moved unchanged from commit
`82341f805ce35e02556a75db9ef174db9d5ceff7`. The 5.1.0 fixture was regenerated
on hosted Windows with Python 3.12.10,
NumPy 2.5.2, SciPy 1.17.1 and PyTorch 2.11.0+cu126. The same job ran the
5.0.0 producer with the same dependencies. The matched comparison found only
the upstream version, the artifact removal contract and the NSGA-III
reference-point annotation changed. The stored-to-matched 5.0.0 comparison
accounts for the floating-point and importance differences in the new fixture.
The added compatibility cases and scikit-learn version were generated in
hosted run 44033529270. Existing cases retain the values generated with
Optuna 5.1.0 in hosted run 44029596220, preserving the existing numerical
expectations across runner differences. The fixture has no embedded generator hash.

| Input | Git blob SHA256 |
| --- | --- |
| `generate_optuna_oracle.py` | `956a6c58750a89e732cfee6fe721fe924714424c35f33ce51cfe8a247e5bc109` |
| `optuna_oracle.json` | `c981660f04e0500ed6fa60284c93ecf49dc09aba9192a88bf66fa8a9eea0e1a7` |

## MATLAB test classification

`optuna_test_manifest.json` is classification metadata, not a numerical oracle.
It was moved byte-for-byte from commit `73ff5ad1b9bee1c5b37fe4e54c20e622580ae59d`
(SHA256 `ff8c3794f3a2f85f7d9e55087df59c16b94d713960a955f77db484bafc2cde43`). The class tables and ordering
are unchanged. The current manifest updates only the renamed oracle
locators. Its package-owned generator requires an explicit test directory
and output; it never discovers a monorepo or rewrites this file implicitly:

```console
python tools/generate_optuna_test_manifest.py --test-directory tests/matlab --output <chosen-file.json>
```

## Installed evidence

`radia_optuna/evidence_manifest.json` names the six JSON inputs and the two
MATLAB compatibility ledgers, with SHA-256 over LF-normalized bytes. It
identifies stored files, not a new generation run or numerical certification.
The distribution manifest links it; wheel verification checks every payload
against both this ledger and the checked package source. No producer is shipped.
Existing fixture provenance and the reviewed identities above remain unchanged.

## Regeneration command

After approval, run from the package directory with the recorded Python
interpreter and a clean, committed package tree. This command uses the two
NumPy environments recorded in `upstream.json` and installs CPU-side oracle
dependencies into scratch venvs. It never builds or launches CUDA kernels.

```console
python tools/regenerate_oracles.py --optuna <version> --optuna-mcp <version> --out <new-scratch-dir> --reason "<approved reason>"
```

Use a directory outside the checkout that does not exist yet. The tool archives
the committed package, adapts only the frozen producer version guards in
scratch, maps historical output names, and runs the five producers and two
derived generators. The proposed files go under `fixtures/`. Review
`provenance.json` for the source, original and executed producer hashes,
interpreter, installed distributions and exact commands. Review `diff.json`
for byte counts, hashes, case paths, sequence and object order, numerical
deltas, and changes under RNG-related fields. Per-command logs and the
snapshot are kept in the output directory for diagnosis. If the producer
source has already been updated for the target version, pass
`--baseline-ref <old-reviewed-commit>` to compare its generated output with
the previous fixtures.

Review and adopt the proposed files explicitly in the migration PR along
with the producer edits, compatibility metadata and fixture identity hashes.
For the CUDA oracle, compare the embedded producer fingerprint against the
actual LF-normalized producer in that PR, then generate again from that exact
committed source if it differs. A generation report establishes provenance
for the run and does not certify any numerical change.

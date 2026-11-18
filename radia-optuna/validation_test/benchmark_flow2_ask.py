"""Flow2 development timing for the pinned sampler and exact native extension."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from itertools import chain
import socket
import subprocess
from time import perf_counter_ns

import numpy as np
import optuna
from optuna.samplers._tpe import probability_distributions as distributions
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator

from validate_kernels import ROOT, descriptor, sha


def native_inputs(samples, mixture):
    rows = np.array(samples, dtype=float, copy=True, order="C")
    specs = []
    for column, d in enumerate(mixture.distributions):
        if isinstance(d, distributions._BatchedCategoricalDistributions):
            rows[:, column] += 1  # Native categorical indices are one-based.
            spec = dict(categorical=True, probabilities=d.weights.tolist())
        else:
            spec = dict(categorical=False, mu=d.mu.tolist(), sigma=d.sigma.tolist(),
                        internal_low=d.adapted_low, internal_high=d.adapted_high,
                        step=d.step or None, log=d.is_log)
        specs.append(descriptor(dict(spec, weights=mixture.weights.tolist())))
    return rows.ravel().tolist(), specs


class TracedSampler(optuna.samplers.TPESampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.winners = []

    def _compute_acquisition_func(self, samples, mpe_below, mpe_above):
        scores = super()._compute_acquisition_func(samples, mpe_below, mpe_above)
        self.winners.append(int(np.argmax(scores)))
        return scores


class NativeScoringSampler(TracedSampler):
    """Benchmark adapter. Upstream owns history, sampling and RNG."""

    def __init__(self, backend, kernels, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.kernels = kernels

    def _compute_acquisition_func(self, samples, mpe_below, mpe_above):
        rows, below = native_inputs(
            mpe_below._transform(samples), mpe_below._mixture_distribution
        )
        _, above = native_inputs(mpe_above._transform(samples), mpe_above._mixture_distribution)
        index = self.kernels.best(self.backend, rows, below, above)
        # Upstream _compare consumes only the argmax, so expose the native winner.
        acquisition = np.zeros(len(next(iter(samples.values()))))
        acquisition[index] = 1
        self.winners.append(index)
        return acquisition


def time_call(function, samples):
    start = perf_counter_ns()
    value = function()
    samples.append(perf_counter_ns() - start)
    return value


def stats_delta(before, after):
    return {key: after[key] - value for key, value in before.items()
            if type(value) is int and type(after[key]) is int}


def ask_case(kernels, backend, candidates, components, calls):
    options = dict(seed=721, n_ei_candidates=candidates, multivariate=True)
    sampler = (TracedSampler(**options) if backend == "numpy"
               else NativeScoringSampler(backend, kernels, **options))
    study = optuna.create_study(sampler=sampler)
    space = {"x": optuna.distributions.FloatDistribution(-2, 2),
             "scale": optuna.distributions.FloatDistribution(1e-3, 1, log=True),
             "step": optuna.distributions.IntDistribution(0, 8, step=2),
             "category": optuna.distributions.CategoricalDistribution(["a", "b", "c"])}
    rng = np.random.default_rng(721)
    for _ in range(components - 1):
        params = dict(x=float(rng.uniform(-2, 2)), scale=float(np.exp(rng.uniform(-6, 0))),
                      step=int(rng.integers(5)) * 2, category=str(rng.choice(["a", "b", "c"])))
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=space, value=params["x"] ** 2 + params["scale"]
        ))
    timings = []
    proposals = []
    indices = []
    for _ in range(calls):
        sampler.winners.clear()
        trial = time_call(lambda: study.ask(fixed_distributions=space), timings)
        proposals.append(trial.params)
        indices.append(list(sampler.winners))
        study.tell(trial, trial.params["x"] ** 2 + trial.params["scale"])
    checksum = hashlib.sha256(json.dumps(proposals, sort_keys=True, allow_nan=False).encode())
    return {"samples_ns": timings, "proposal_sha256": checksum.hexdigest(),
            "proposals": proposals, "selected_indices": indices}


def primitive_case(kernels, backend, candidates, components, calls):
    rng = np.random.default_rng(721)
    name = "x"
    space = {name: optuna.distributions.FloatDistribution(-2, 2)}
    settings = optuna.samplers.TPESampler()._parzen_estimator_parameters
    observations = [rng.uniform(a, b, components - 1) for a, b in [(-2, 1), (-1, 2)]]

    def estimate(values):
        return _ParzenEstimator({name: values}, space, settings)._mixture_distribution

    below, above = map(estimate, observations)
    values = rng.uniform(-2, 2, size=(candidates, 1))
    ref = int(np.argmax(below.log_pdf(values) - above.log_pdf(values)))
    timings = {"log_pdf": [], "best": [], "build_and_best": []}
    for _ in range(calls):
        if backend == "numpy":
            time_call(lambda: below.log_pdf(values), timings["log_pdf"])
            best = time_call(lambda: int(np.argmax(below.log_pdf(values) - above.log_pdf(values))),
                             timings["best"])
        else:
            def density():
                rows, spec = native_inputs(values, below)
                return kernels.log_pdf(backend, rows, spec)

            def select(lo, hi):
                rows, lower = native_inputs(values, lo)
                _, upper = native_inputs(values, hi)
                return kernels.best(backend, rows, lower, upper)

            time_call(density, timings["log_pdf"])
            best = time_call(lambda: select(below, above), timings["best"])

        def build_and_best():
            if backend == "gpu":
                dim = kernels.Dimension()
                dim.low, dim.high, dim.step = -2, 2, float("nan")
                dim.minimum = 4 / min(100, 1 + components)
                dim.choices, dim.categorical, dim.log_scale = 0, False, False
                models = []
                try:
                    for data, mix in zip(observations, (below, above)):
                        models.append(kernels.build_model(
                            backend, data.tolist(), mix.weights.tolist(), [dim], 1.0,
                            settings.consider_endpoints, False
                        ))
                    return kernels.best_prepared(backend, values.ravel().tolist(), *models)
                finally:
                    for model in models:
                        model.close()
            lo, hi = map(estimate, observations)
            if backend == "cpu":
                return select(lo, hi)
            return int(np.argmax(lo.log_pdf(values) - hi.log_pdf(values)))

        prepared = time_call(build_and_best, timings["build_and_best"])
        if best != ref or prepared != ref:
            raise RuntimeError(f"Primitive selection mismatch: {best=}, {prepared=}, {ref=}")
    return {"samples_ns": timings, "selected_index": ref}


def fixture_mixture(specs):
    batch = []
    for d in specs:
        if d["categorical"]:
            batch.append(distributions._BatchedCategoricalDistributions(np.array(d["probabilities"])))
            continue
        args = (np.array(d["mu"]), np.array(d["sigma"]), d["low"], d["high"])
        if d["step"] is not None:
            cls = (distributions._BatchedDiscreteTruncLogNormDistributions if d["log"]
                   else distributions._BatchedDiscreteTruncNormDistributions)
            batch.append(cls(*args, d["step"]))
        else:
            cls = (distributions._BatchedTruncLogNormDistributions if d["log"]
                   else distributions._BatchedTruncNormDistributions)
            batch.append(cls(*args))
    return distributions._MixtureOfProductDistribution(np.array(specs[0]["weights"]), batch)


def formal_cases(kernels, backend, calls):
    oracle = ROOT / "oracles/optuna50_cuda_oracle.json"
    data = json.loads(oracle.read_text(encoding="utf-8"))
    if data["optuna_version"] != optuna.__version__:
        raise RuntimeError("Wrong stored oracle version")
    for case in data["cases"]:
        values = np.asarray(case["samples"])
        if backend == "numpy":
            lo, hi = (fixture_mixture(case[key]) for key in ("below", "above"))
            for column, d in enumerate(case["below"]):
                if d["categorical"]:
                    values[:, column] -= 1
            log_below, log_above = lo.log_pdf, hi.log_pdf

            def best_candidate():
                return int(np.argmax(log_below(values) - log_above(values)))
        else:
            rows = values.ravel().tolist()
            below, above = ([descriptor(d) for d in case[key]] for key in ("below", "above"))

            def log_below(_values):
                return kernels.log_pdf(backend, rows, below)

            def log_above(_values):
                return kernels.log_pdf(backend, rows, above)

            def best_candidate():
                return kernels.best(backend, rows, below, above)

            # Cross-lane and cross-stride ties must select the smallest index.
            tied = rows[:len(below)] * 257
            if kernels.best(backend, tied, below, below) != 0:
                raise RuntimeError("First-maximum tie changed")
        timings = {"log_pdf": [], "best": []}
        for _ in range(calls):
            scores = time_call(lambda: log_below(values), timings["log_pdf"])
            best = time_call(best_candidate, timings["best"])
            np.testing.assert_allclose(scores, case["below_pdf"], rtol=0, atol=1e-10)
            np.testing.assert_allclose(log_above(values), case["above_pdf"], rtol=0, atol=1e-10)
            if best + 1 != case["best_index"]:
                raise RuntimeError(f"Stored selection mismatch: {case['name']}")
        yield {"case": case["name"], "backend": backend, "samples_ns": timings}


def formal_models(kernels, backend, calls):
    oracle = ROOT / "oracles/optuna50_cuda_oracle.json"
    data = json.loads(oracle.read_text(encoding="utf-8"))
    space = {
        "x": optuna.distributions.FloatDistribution(-2, 2),
        "log": optuna.distributions.FloatDistribution(float(np.exp(-2)), float(np.exp(2)), log=True),
        "integer": optuna.distributions.IntDistribution(0, 4096),
        "category": optuna.distributions.CategoricalDistribution(["a", "b", "c"]),
    }
    defaults = optuna.samplers.TPESampler()._parzen_estimator_parameters
    for case in data["model_cases"]:
        size = len(case["good"])
        raw = [np.asarray(case[key]).reshape(size, len(space)) for key in ("good", "bad")]
        settings = defaults._replace(consider_endpoints=case["endpoints"], multivariate=True)

        def fit(values):
            observations = {key: values[:, i] - (key == "category") for i, key in enumerate(space)}
            return _ParzenEstimator(observations, space, settings)

        lower, upper = map(fit, raw)
        rng = np.random.RandomState(case["seed"])
        samples = lower.sample(rng, 24)
        np.testing.assert_array_equal(rng.rand(8), case["next_random"])
        matrix = lower._transform(samples)
        native_rows, _ = native_inputs(matrix, lower._mixture_distribution)
        native_matrix = np.asarray(native_rows).reshape(-1, len(space))
        expected_index = int(np.argmax(lower.log_pdf(samples) - upper.log_pdf(samples)))
        # Use the existing MATLAB oracle tolerance for platform-specific RNG math.
        np.testing.assert_allclose(native_matrix[expected_index], case["best"], rtol=0, atol=5e-11)

        def build_and_best():
            if backend == "gpu":
                models = []
                try:
                    for values, estimator in zip(raw, (lower, upper)):
                        mixture = estimator._mixture_distribution
                        dims = []
                        for dist in mixture.distributions:
                            dim = kernels.Dimension()
                            dim.categorical = isinstance(dist, distributions._BatchedCategoricalDistributions)
                            if dim.categorical:
                                dim.low, dim.high, dim.choices = 0, 1, dist.weights.shape[1]
                                dim.minimum, dim.step, dim.log_scale = 0, float("nan"), False
                            else:
                                dim.low, dim.high = dist.adapted_low, dist.adapted_high
                                dim.step, dim.log_scale, dim.choices = dist.step or float("nan"), dist.is_log, 0
                                dim.minimum = (dim.high - dim.low) / min(100, size + 2)
                            dims.append(dim)
                        models.append(kernels.build_model(
                            backend, values.ravel().tolist(), mixture.weights.tolist(),
                            dims, 1.0, case["endpoints"], False
                        ))
                    return kernels.best_prepared(backend, native_rows, *models)
                finally:
                    for model in models:
                        model.close()
            lo, hi = map(fit, raw)
            if backend == "cpu":
                _, lower_spec = native_inputs(matrix, lo._mixture_distribution)
                _, upper_spec = native_inputs(matrix, hi._mixture_distribution)
                return kernels.best(backend, native_rows, lower_spec, upper_spec)
            return int(np.argmax(lo.log_pdf(samples) - hi.log_pdf(samples)))

        durations = []
        for _ in range(calls):
            best = time_call(build_and_best, durations)
            if best != expected_index:
                raise RuntimeError(
                    f"Stored model case {case['name']} ({backend}): "
                    f"candidate {best} != upstream candidate {expected_index}"
                )
        yield {"case": case["name"], "backend": backend, "samples_ns": {"build_and_best": durations}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=("primitives", "ask"))
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--colocated-job", action="append", default=[])
    parser.add_argument("--calls", type=int, default=12)
    parser.add_argument("--installed-wheel", action="store_true",
                        help="require the native module from an installed wheel, not this checkout")
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("--calls must be positive")

    from radia_optuna.kernels import load

    pin = json.loads((ROOT / "src/radia_optuna/upstream.json").read_text())["optuna"]
    if optuna.__version__ != pin:
        raise RuntimeError(f"Optuna pin mismatch: {optuna.__version__} != {pin}")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    receipt = json.loads(args.build_receipt.read_text())
    kernels = load("cpu")
    binary = Path(kernels.__file__).resolve()
    if args.installed_wheel and ROOT.resolve() in binary.parents:
        raise RuntimeError(f"Native module resolved inside the checkout: {binary}")
    binary_sha = sha(binary)
    if commit != receipt["source_sha"] or binary_sha != receipt["python_extension_sha256s"].get(binary.name):
        raise RuntimeError("Native build receipt mismatch")
    report = dict(
        lane=args.lane, source_sha=commit, extension_sha256=binary_sha,
        benchmark_sha256=sha(Path(__file__)), build_receipt_sha256=sha(args.build_receipt),
        oracle_sha256=sha(ROOT / "oracles/optuna50_cuda_oracle.json"),
        node=socket.gethostname(), affinity=sorted(os.sched_getaffinity(0)),
        allocation=os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"),
        gpu=subprocess.check_output(["nvidia-smi", "-L"], text=True).strip(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        omp_threads=os.environ.get("OMP_NUM_THREADS"), colocated_jobs=args.colocated_job,
        optuna=optuna.__version__, numpy=np.__version__, cases=[], ok=False,
    )
    try:
        case_runner = primitive_case if args.lane == "primitives" else ask_case
        for candidates, components in ((24, 32), (24, 256), (256, 256), (4096, 512)):
            reference = None
            for backend in ("numpy", "cpu", "gpu"):
                before = kernels.statistics()
                record = case_runner(kernels, backend, candidates, components, args.calls)
                report["cases"].append(dict(
                    record, candidates=candidates, initial_components=components, backend=backend,
                    statistics_delta=stats_delta(before, kernels.statistics()),
                ))
                if args.lane == "ask":
                    digest = record["proposal_sha256"]
                    if backend == "numpy":
                        reference = record
                    elif digest != reference["proposal_sha256"] or record["selected_indices"] != reference["selected_indices"]:
                        for i, (expected, actual) in enumerate(zip(reference["proposals"], record["proposals"])):
                            lo, hi = reference["selected_indices"][i], record["selected_indices"][i]
                            if expected != actual or lo != hi:
                                report["first_proposal_mismatch"] = dict(
                                    backend=backend, candidates=candidates, initial_components=components,
                                    call=i + 1, upstream_indices=lo, native_indices=hi,
                                )
                                break
                        raise RuntimeError(f"Seeded proposal mismatch: {backend}, {candidates}, {components}")
        if args.lane == "primitives":
            for backend in ("numpy", "cpu", "gpu"):
                before = kernels.statistics()
                for record in chain(formal_cases(kernels, backend, args.calls),
                                    formal_models(kernels, backend, args.calls)):
                    after = kernels.statistics()
                    report["cases"].append(dict(record, statistics_delta=stats_delta(before, after)))
                    before = after
        report["ok"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8", newline="\n")
        kernels.shutdown()


if __name__ == "__main__":
    main()

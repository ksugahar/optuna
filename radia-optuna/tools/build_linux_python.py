"""Build the Linux Python binding with an explicitly selected CUDA toolchain."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--cxx", type=Path, required=True)
    parser.add_argument("--cuda-architectures", required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this entry point requires Linux")
    for compiler in (args.nvcc, args.cxx):
        if not compiler.is_absolute() or not compiler.is_file():
            parser.error(f"an existing absolute compiler path is required: {compiler}")
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=False)
    pybind = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--cmakedir"], text=True
    ).strip()
    configure = [
        "cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={args.cxx}",
        f"-DCMAKE_CUDA_COMPILER={args.nvcc}",
        f"-DCMAKE_CUDA_HOST_COMPILER={args.cxx}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_architectures}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind}",
        "-DOPTUNA_BUILD_MATLAB=OFF",
    ]
    build_command = ["cmake", "--build", str(build), "--parallel", "2", "--target", "_optuna_kernels"]
    subprocess.run(configure, check=True)
    subprocess.run(build_command, check=True)
    binary = ROOT / "src/radia_optuna" / ("_optuna_kernels" + sysconfig.get_config_var("EXT_SUFFIX"))
    source = sorted(p for p in (ROOT / "native").iterdir() if p.is_file())
    recipes = [ROOT / "CMakeLists.txt", Path(__file__).resolve()]
    receipt = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "recipe_hashes": {p.relative_to(ROOT).as_posix(): sha256(p) for p in recipes},
        "source_hashes": {p.relative_to(ROOT).as_posix(): sha256(p) for p in source},
        "archive_sha256": sha256(build / "liboptuna_kernels.a"),
        "python_extension_sha256s": {binary.name: sha256(binary)},
        "commands": [configure, build_command],
    }
    (build / "python-build.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()

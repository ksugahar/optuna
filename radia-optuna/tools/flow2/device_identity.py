"""Record and check the allocated GPU before a Flow2 GPU validation.

Fails unless exactly one GPU is visible and it matches the expected name and
compute capability. The record is the device identity part of the result JSON.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-name", required=True, help="substring of the GPU name, e.g. GB200")
    parser.add_argument("--expect-compute-cap", required=True, help="e.g. 10.0")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    fields = "index,uuid,name,compute_cap,driver_version,mig.mode.current,memory.total"
    rows = [
        [cell.strip() for cell in line.split(",")]
        for line in subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"], text=True
        ).strip().splitlines()
    ]
    record = {
        "schema": "radia.optuna.flow2-device-identity.v1",
        "node": socket.gethostname(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "affinity": sorted(os.sched_getaffinity(0)),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "gpus": [dict(zip(fields.split(","), row)) for row in rows],
        "nvidia_smi_list": subprocess.check_output(["nvidia-smi", "-L"], text=True).strip(),
    }
    args.report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one visible GPU, found {len(rows)}")
    gpu = record["gpus"][0]
    if args.expect_name not in gpu["name"] or gpu["compute_cap"] != args.expect_compute_cap:
        raise SystemExit(
            f"unexpected GPU: {gpu['name']} CC {gpu['compute_cap']}; "
            f"expected {args.expect_name} CC {args.expect_compute_cap}"
        )


if __name__ == "__main__":
    main()

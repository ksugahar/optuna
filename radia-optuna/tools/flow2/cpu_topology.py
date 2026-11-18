"""Record the Type I node topology and choose pinned cores for each thread count.

One hardware thread per physical core, filled socket by socket and NUMA node by
NUMA node in CPU order, so that N threads use the fewest sockets and nodes. The
report lists, for every thread count, the CPUs, physical cores, sockets and
NUMA nodes used; type1.pbs pins each timing run to exactly that CPU list.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess

SYS = Path("/sys/devices/system")


def cpu_record(cpu):
    base = SYS / f"cpu/cpu{cpu}"
    nodes = [int(p.name[4:]) for p in base.iterdir() if re.fullmatch(r"node\d+", p.name)]
    if len(nodes) != 1:
        raise RuntimeError(f"cpu{cpu}: expected one NUMA node, found {nodes}")
    topology = base / "topology"
    return {"cpu": cpu, "node": nodes[0],
            "socket": int((topology / "physical_package_id").read_text()),
            "core": int((topology / "core_id").read_text())}


def select(records, threads):
    first = {}
    for record in sorted(records, key=lambda r: r["cpu"]):
        first.setdefault((record["socket"], record["core"]), record)
    cores = sorted(first.values(), key=lambda r: (r["socket"], r["node"], r["cpu"]))
    if len(cores) < threads:
        raise RuntimeError(f"{threads} threads need {threads} physical cores; only {len(cores)} allowed")
    chosen = cores[:threads]
    return {"cpus": [r["cpu"] for r in chosen],
            "sockets": sorted({r["socket"] for r in chosen}),
            "numa_nodes": sorted({r["node"] for r in chosen})}


def output(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    records = [cpu_record(cpu) for cpu in allowed]
    report = {
        "schema": "radia.optuna.flow2-cpu-topology.v1",
        "node": socket.gethostname(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "allowed_cpus": allowed,
        "physical_cores": len({(r["socket"], r["core"]) for r in records}),
        "sockets": sorted({r["socket"] for r in records}),
        "numa_nodes": sorted({r["node"] for r in records}),
        "lscpu": json.loads(output(["lscpu", "-J"])),
        "numactl_hardware": output(["numactl", "--hardware"]) if shutil.which("numactl") else None,
        "selections": {str(n): select(records, n) for n in args.threads},
        "policy": "one hardware thread per physical core, socket then NUMA node then CPU order",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

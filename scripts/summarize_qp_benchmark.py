"""Read current experiment artifacts and print compact, paired factual results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sqlite3


def nsight_summary(path: Path) -> dict:
    tables = []
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("Time (%),"):
            if rows:
                tables.append(rows)
            rows = []
        elif rows or line[:1].isdigit():
            if not line:
                if rows:
                    tables.append(rows)
                    rows = []
            else:
                rows.append(next(csv.reader(io.StringIO(line))))
    if rows:
        tables.append(rows)
    if len(tables) != 3 or not tables[0]:
        raise ValueError(f"incomplete Nsight tables: {path}")
    return {
        "kernel_total_s": sum(float(row[1]) for row in tables[0]) * 1e-9,
        "kernel_instances": sum(int(row[2]) for row in tables[0]),
        "top_kernels": [{"name": row[-1], "total_s": float(row[1]) * 1e-9} for row in tables[0][:3]],
        "api_total_s": sum(float(row[1]) for row in tables[1]) * 1e-9,
        "top_api": [{"name": row[-1], "calls": int(row[2]), "total_s": float(row[1]) * 1e-9} for row in tables[1][:3]],
    }


def summary(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if raw["verdict"] != "PASS":
        raise ValueError(f"failed experiment cannot be summarized as accepted: {path}")
    item = {
        "source": str(path), "at": raw["finished_at"], "hash": raw["problem_hash"],
        "n": raw["dimensions"]["variables"], "hardware": raw["hardware"],
        "software": raw["software"],
        "commercial": raw["commercial"], "references": {},
    }
    for solver, result in raw["references"].items():
        item["references"][solver] = {
            mode: values["latency"] for mode, values in result.items()
            if isinstance(values, dict) and "latency" in values
        } or result
    if "openopt" not in raw:
        return item
    item["latencies"] = {
        device: {mode: result["latency"] for mode, result in values.items()}
        for device, values in raw["openopt"].items()
    }
    qualities = [
        q for device in raw["openopt"].values()
        for mode in device.values() for q in mode["quality"]
    ]
    if any(q["verdict"] != "PASS" for q in qualities) or not qualities:
        raise ValueError("quality artifacts incomplete")
    item["max_quality"] = {
        key: max(q[key] for q in qualities) for key in (
            "primal_residual", "dual_residual", "normal_cone_residual",
            "relative_objective_gap", "solution_error",
        )
    }
    item["speedups"] = {
        mode: item["latencies"]["cpu"][mode]["p50_s"] / item["latencies"]["cuda:0"][mode]["p50_s"]
        for mode in ("cold", "prepared")
    }
    if "paired_devices" in raw:
        item["paired_devices"] = {}
        for mode, devices in raw["paired_devices"].items():
            cpu = devices["cpu"]["latency"]
            gpu = devices["cuda:0"]["latency"]
            item["paired_devices"][mode] = {
                "cpu": cpu, "gpu": gpu, "speedup_p50": cpu["p50_s"] / gpu["p50_s"],
                "paired_speedups": [a / b for a, b in zip(cpu["samples_s"], gpu["samples_s"], strict=True)],
            }
    if "paired_ab" in raw:
        item["paired_ab"] = {}
        for mode, arms in raw["paired_ab"].items():
            base = arms["baseline"]["latency"]["p50_s"]
            candidate = arms["triangular"]["latency"]["p50_s"]
            control = arms["identical_control"]["latency"]["p50_s"]
            floor = abs(base - control)
            item["paired_ab"][mode] = {
                "baseline_p50_s": base, "triangular_p50_s": candidate,
                "control_p50_s": control, "noise_floor_s": floor,
                "speedup": base / candidate,
                "measured_benefit_above_floor": base - candidate > floor,
            }
    samples = json.loads((path.parent / "telemetry.json").read_text())
    if not samples or any(s["rc"] for s in samples):
        raise ValueError("telemetry is missing or failed")
    gpu = [s["gpu"].split(",") for s in samples]
    item["telemetry"] = {
        "samples": len(samples), "gpu_utilization_max_pct": max(float(row[1]) for row in gpu),
        "gpu_utilization_mean_pct": sum(float(row[1]) for row in gpu) / len(gpu),
        "gpu_memory_max_mib": max(float(row[2]) for row in gpu),
        "cpu_usage_delta_s": (samples[-1]["cpu"]["usage_usec"] - samples[0]["cpu"]["usage_usec"]) / 1e6,
        "window_s": samples[-1]["monotonic_s"] - samples[0]["monotonic_s"],
    }
    item["nsight"] = nsight_summary(path.parent / "nsys-stats.csv")
    if (path.parent / "profile-wall.json").is_file():
        wall = json.loads((path.parent / "profile-wall.json").read_text())["wall_s"]
        item["nsight"]["profiled_prepared_e2e_s"] = wall
        item["nsight"]["kernel_over_profiled_e2e"] = item["nsight"]["kernel_total_s"] / wall
    db_path = path.parent / "nsight.sqlite"
    if db_path.is_file():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            start, end = db.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
            if start is None or end is None:
                raise ValueError("Nsight kernel events empty")
            item["nsight"]["kernel_span_s"] = (end - start) * 1e-9
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    items = [summary(path) for path in args.results]
    result = {"read_at": datetime.now(timezone.utc).isoformat(), "items": items}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for item in items:
        print(f"QP_CURRENT n={item['n']} at={item['at']} hash={item['hash']}")
        if "speedups" in item:
            for mode, ratio in item["speedups"].items():
                cpu = item["latencies"]["cpu"][mode]["p50_s"] * 1000
                gpu = item["latencies"]["cuda:0"][mode]["p50_s"] * 1000
                print(f"LATENCY mode={mode} CPU_ms={cpu:.4f} GPU_ms={gpu:.4f} speedup={ratio:.3f}")
            for mode, values in item.get("paired_ab", {}).items():
                print(f"AB mode={mode} {json.dumps(values)}")
            for mode, values in item.get("paired_devices", {}).items():
                print(f"PAIRED mode={mode} CPU_ms={values['cpu']['p50_s']*1000:.4f} "
                      f"GPU_ms={values['gpu']['p50_s']*1000:.4f} speedup={values['speedup_p50']:.3f}")
            print(f"QUALITY {json.dumps(item['max_quality'])}")
            print(f"RESOURCES {json.dumps(item['telemetry'])}")
            print(f"NSIGHT {json.dumps(item['nsight'])}")
        else:
            print(f"REFERENCES {json.dumps(item['references'])}")
    print(f"QP_CURRENT verdict=PASS read_at={result['read_at']} cases={len(items)}")


if __name__ == "__main__":
    main()

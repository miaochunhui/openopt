from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch


TARGET_ARCHITECTURES = {
    "NVIDIA A100 40GB": "sm_80",
    "NVIDIA GeForce RTX 3080": "sm_86",
    "NVIDIA RTX PRO 4500 Blackwell": "sm_120",
}

TARGET_MATCHERS = {
    "NVIDIA A100 40GB": {"name_contains": "A100", "minimum_memory_gib": 38.0},
    "NVIDIA GeForce RTX 3080": {"name_contains": "RTX 3080", "minimum_memory_gib": 9.0},
    "NVIDIA RTX PRO 4500 Blackwell": {
        "name_contains": "RTX PRO 4500",
        "minimum_memory_gib": 30.0,
    },
}


def compatibility_report() -> dict[str, object]:
    compiled_architectures = torch.cuda.get_arch_list()
    visible_devices = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
            "architecture": "sm_" + "".join(map(str, torch.cuda.get_device_capability(index))),
            "total_memory_gib": torch.cuda.get_device_properties(index).total_memory / (1024**3),
        }
        for index in range(torch.cuda.device_count())
    ]
    targets: dict[str, object] = {}
    for name, architecture in TARGET_ARCHITECTURES.items():
        matcher = TARGET_MATCHERS[name]
        matching_devices = [
            device
            for device in visible_devices
            if matcher["name_contains"] in str(device["name"])
            and device["architecture"] == architecture
            and float(device["total_memory_gib"]) >= float(matcher["minimum_memory_gib"])
        ]
        build_supported = architecture in compiled_architectures
        runtime_verified = build_supported and bool(matching_devices)
        if not build_supported:
            status = "UNSUPPORTED_BUILD"
            reason = f"{architecture} is absent from torch.cuda.get_arch_list()"
        elif runtime_verified:
            status = "VERIFIED_LOCAL"
            reason = "matching hardware is visible and its architecture is present in the torch build"
        else:
            status = "UNVERIFIED_NO_HARDWARE"
            reason = "architecture is present in the torch build, but matching hardware is not visible"
        targets[name] = {
            "architecture": architecture,
            "present_in_torch_build": build_supported,
            "runtime_verified": runtime_verified,
            "verification_status": status,
            "matching_device_indices": [int(device["index"]) for device in matching_devices],
            "reason": reason,
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "compiled_architectures": compiled_architectures,
        "targets": targets,
        "visible_devices": visible_devices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    serialized = json.dumps(compatibility_report(), indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

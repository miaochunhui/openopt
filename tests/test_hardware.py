from __future__ import annotations

from types import SimpleNamespace

from openpo import hardware


def test_compatibility_report_separates_build_and_runtime_verification(monkeypatch) -> None:
    monkeypatch.setattr(hardware.torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_86", "sm_120"])
    monkeypatch.setattr(hardware.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(hardware.torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 3080")
    monkeypatch.setattr(hardware.torch.cuda, "get_device_capability", lambda _: (8, 6))
    monkeypatch.setattr(
        hardware.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=10 * 1024**3),
    )

    report = hardware.compatibility_report()

    assert report["targets"]["NVIDIA GeForce RTX 3080"]["verification_status"] == "VERIFIED_LOCAL"
    assert report["targets"]["NVIDIA A100 40GB"]["present_in_torch_build"] is True
    assert report["targets"]["NVIDIA A100 40GB"]["runtime_verified"] is False
    assert (
        report["targets"]["NVIDIA RTX PRO 4500 Blackwell"]["verification_status"]
        == "UNVERIFIED_NO_HARDWARE"
    )


def test_compatibility_report_marks_missing_architecture_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(hardware.torch.cuda, "get_arch_list", lambda: ["sm_86"])
    monkeypatch.setattr(hardware.torch.cuda, "device_count", lambda: 0)

    report = hardware.compatibility_report()

    assert report["targets"]["NVIDIA A100 40GB"]["verification_status"] == "UNSUPPORTED_BUILD"
    assert report["targets"]["NVIDIA RTX PRO 4500 Blackwell"]["present_in_torch_build"] is False

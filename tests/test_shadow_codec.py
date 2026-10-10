from __future__ import annotations

import json

import pytest
import torch

from openpo.facade import input_sha256, solve_production
from openpo.shadow_codec import (
    decode_payload,
    encode_payload,
    production_from_payload,
    production_to_payload,
    request_from_payload,
    request_to_payload,
)
from tests.test_facade import _request


def test_durable_request_codec_is_deterministic_and_roundtrips() -> None:
    request = _request(exact_backend="osqp")
    first = encode_payload(request_to_payload(request))
    second = encode_payload(request_to_payload(request))
    restored = request_from_payload(decode_payload(first))

    assert first == second
    assert input_sha256(restored.problem, restored.constraints) == input_sha256(
        request.problem,
        request.constraints,
    )
    assert restored.fast_device == request.fast_device
    assert restored.exact_backend == request.exact_backend
    assert restored.fast_config == request.fast_config


def test_durable_production_result_codec_roundtrips_evidence() -> None:
    result = solve_production(_request(exact_backend="osqp"))
    encoded = encode_payload(production_to_payload(result))
    restored = production_from_payload(decode_payload(encoded))

    assert restored.result_sha256 == result.result_sha256
    assert restored.fast.result_sha256 == result.fast.result_sha256
    assert restored.exact.result_sha256 == result.exact.result_sha256
    assert restored.fast.residuals == result.fast.residuals
    assert torch.equal(restored.fast.weights, result.fast.weights.cpu())
    assert torch.equal(restored.exact.objective, result.exact.objective.cpu())


def test_durable_codec_rejects_tensor_corruption() -> None:
    payload = request_to_payload(_request(exact_backend="osqp"))
    payload["problem"]["alpha"]["sha256"] = "f" * 64

    with pytest.raises(ValueError, match="tensor hash mismatch"):
        request_from_payload(json.loads(encode_payload(payload)))


def test_durable_codec_rejects_pickle_or_non_json_payload() -> None:
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        decode_payload(b"\x80\x04pickle")

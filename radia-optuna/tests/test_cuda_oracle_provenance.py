"""Package-owned upstream oracle identity contract.

Guards: Changes to the approved producer or stored oracle expectations, drift of
its upstream pin, and reintroducing monorepo-relative verification inputs.
Assumes: Stored upstream Optuna evidence; producer identity uses its
existing LF normalization. This checks source identity, not solver accuracy.
"""

import hashlib
import json
from pathlib import Path
import tomllib

import pytest


@pytest.mark.parametrize(
    "producer_name,fixture_name,expected_producer,expected_fixture,embedded_hash",
    [
        pytest.param(
            "generate_optuna50_cuda_oracle.py", "optuna50_cuda_oracle.json",
            "b5e49a9bea5253c8e1a7b55222a3a2a66a3915da475797c5d1248f8ecb0dd7dc",
            "73375a7e511cfccaa4916b4304c433c9f53e3d881df0b10c9dc5a3c29d0fa4d8",
            True, id="cuda50",
        ),
        pytest.param(
            "generate_optuna_cuda_density_oracle.py", "optuna_cuda_density_oracle.json",
            "3cf68642bd20dd9d1eafe32f659525689cdc26843f7fbbc83a1bb2f1e5e49b36",
            "167f8bef1d1088ab5c55c6bb2845c97e6084a496e665d672e9e877b20697545b",
            False, id="density",
        ),
        pytest.param(
            "generate_optuna_mcp_oracle.py", "optuna_mcp_oracle.json",
            "7d6a08a91e8fe21a438660e6f359bcfa468578efd9c7579c6e592dbec56dd6cb",
            "890bd0c25e6842427db8c55a6c1cda4135afdacb03ea2d53b5c4785aa15aa8aa",
            False, id="mcp50",
        ),
        pytest.param(
            "generate_optuna_api_inventory.py", "optuna_public_api.json",
            "bcc580d912b0e5f724d8e9ece58d48674f267a208a792ffe535fabdce81ece12",
            "f4b8bd46dd72bcafc8bd79651f368d90e7d77eeb318ad765f30b2363137bf678",
            False, id="api50",
        ),
        pytest.param(
            "generate_optuna_oracle.py", "optuna_oracle.json",
            "4524b64b6f9bfac45bd0abd24fd6633ee50de7780920cb942341dcd6613dfdac",
            "89bb7a0f907ef7158213deca5d2e2feb7810b95191c5d96bcb6bfed456183264",
            False, id="upstream50",
        ),
    ],
)
def test_cuda_oracle_preserves_approved_source_identity(
    producer_name, fixture_name, expected_producer, expected_fixture, embedded_hash,
):
    package = Path(__file__).resolve().parents[1]
    inputs = package / "oracles"
    producer = (inputs / producer_name).read_bytes()
    producer_hash = hashlib.sha256(producer.replace(b"\r\n", b"\n")).hexdigest()
    assert producer_hash == expected_producer, (producer_hash, expected_producer)
    fixture = (inputs / fixture_name).read_bytes()
    fixture_hash = hashlib.sha256(fixture).hexdigest()
    assert fixture_hash == expected_fixture, (fixture_hash, expected_fixture)
    oracle = json.loads(fixture)
    if embedded_hash:
        assert oracle["generator_sha256"] == producer_hash, oracle["generator_sha256"]
    else:
        # The unchanged legacy schema has no embedded producer fingerprint.
        assert "generator_sha256" not in oracle, oracle.keys()
    pin = json.loads((package / "src/radia_optuna/upstream.json").read_text(encoding="utf-8"))
    assert oracle["optuna_version"] == pin["optuna"], oracle["optuna_version"]
    project = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))
    upstream = project["project"]["optional-dependencies"]["upstream"]
    assert f"optuna=={oracle['optuna_version']}" in upstream, upstream

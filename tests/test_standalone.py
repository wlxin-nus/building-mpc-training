"""Regression checks for the export boundary, not new control experiments."""

from __future__ import annotations

import hashlib
import importlib
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from h3c.runtime import clients
from h3c_baselines import cli
from h3c_baselines.mpc.registry import verify_frozen_mpc_suite

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c"


def test_exported_core_and_models_match_provenance() -> None:
    manifest = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for row in manifest["verbatim_files"]:
        path = ROOT / row["path"]
        content = (
            path.read_bytes()
            if row["encoding"] == "bytes"
            else path.read_text(encoding="utf-8").encode()
        )
        assert hashlib.sha256(content).hexdigest() == row["sha256"], row["path"]


def test_imports_resolve_inside_standalone_checkout() -> None:
    for name in (
        "h3c",
        "h3c.runtime.clients",
        "h3c_baselines.mpc.training",
        "h3c_baselines.mpc.registry",
    ):
        module = importlib.import_module(name)
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(ROOT / "src")
    assert not hasattr(clients, "OpenAICompatibleModelClient")
    for excluded in ("agents", "offline", "memory", "diagnostics"):
        assert not (ROOT / "src" / "h3c" / excluded).exists()


def test_frozen_three_case_suite_is_portable() -> None:
    result = verify_frozen_mpc_suite(ROOT / "models" / "mpc_releases" / RELEASE)
    assert result["valid"] is True
    assert result["classification"] == "METHOD-DEGRADED"
    assert set(result["case_checks"]) == {"SZ_Air", "MZ_Hydro", "MZ_Air"}


def test_train_dry_plan_has_no_runtime_side_effects(capsys: pytest.CaptureFixture[str]) -> None:
    before = set((ROOT / "outputs").rglob("*"))
    assert cli.main(["mpc", "train"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["execution"] is False
    assert result["workers"] == 4
    assert set((ROOT / "outputs").rglob("*")) == before


def test_console_entry_preserves_mpc_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["building-mpc", "train"])
    assert cli.mpc_main() == 0
    assert json.loads(capsys.readouterr().out)["execution"] is False


def test_execute_requires_preflight_before_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOPTEST_URL", "http://localhost:5000")

    def blocked(endpoint: str) -> dict[str, Any]:
        raise RuntimeError("blocked before test selection")

    monkeypatch.setattr(cli, "_physical_campaign_preflight", blocked)
    monkeypatch.setattr(
        cli, "train_hierarchical_mpc", lambda **kwargs: pytest.fail("physics before preflight")
    )
    with pytest.raises(RuntimeError, match="blocked before test selection"):
        cli.main(["mpc", "train", "--execute"])


def test_network_preflight_failure_does_not_select_a_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "committed_source_identity", lambda root: "a" * 40)
    monkeypatch.setattr(
        clients.BoptestHttpClient,
        "select_testcase",
        lambda *args: pytest.fail("selected test in TCP preflight"),
    )

    def refused(*args: object, **kwargs: object) -> None:
        raise PermissionError("simulated restricted network")

    monkeypatch.setattr(socket, "create_connection", refused)
    with pytest.raises(PermissionError, match="restricted network"):
        cli._physical_campaign_preflight("http://localhost:5000")


def test_boptest_http_serialization_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def request(method: str, url: str, **kwargs: Any) -> dict[str, Any] | str:
        calls.append((method, url))
        if "/select" in url:
            return {"testid": "fresh-test"}
        if "/status/" in url:
            return "Running"
        return {}

    monkeypatch.setattr(clients, "_request_json", request)
    client = clients.BoptestHttpClient("http://localhost:5000")
    assert client.select_testcase("testcase") == "fresh-test"
    client.stop()
    assert client.test_id is None
    assert calls[-1] == ("PUT", "http://localhost:5000/stop/fresh-test")
    assert json.loads(clients._request_body({"step": 900})) == {"step": 900}
    with pytest.raises(clients.TransportError):
        client.stop()

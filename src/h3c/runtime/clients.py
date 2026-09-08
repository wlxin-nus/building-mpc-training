"""Unchanged physical-only definitions extracted from the upstream module."""

from __future__ import annotations

import http.client
import json
import math
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal, overload


class TransportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        error_type: str = "transport_contract_error",
        provider_response_received: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.error_type = error_type
        self.provider_response_received = provider_response_received
        self.retry_after_seconds = retry_after_seconds


def model_request_body(request_contract: Mapping[str, Any]) -> str:
    """Serialize the exact secret-free model request body sent on the wire."""
    return json.dumps(request_contract, allow_nan=False)


def _request_body(payload: Mapping[str, Any]) -> bytes:
    return model_request_body(payload).encode("utf-8")


def _connection_failure(error: BaseException) -> tuple[bool, str]:
    cause: BaseException | object = error
    if isinstance(error, urllib.error.URLError):
        cause = error.reason
    retryable = isinstance(
        cause,
        (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            TimeoutError,
            socket.gaierror,
            http.client.IncompleteRead,
            ssl.SSLEOFError,
            ssl.SSLZeroReturnError,
        ),
    )
    return retryable, type(cause).__name__


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    try:
        seconds = float(stripped)
    except ValueError:
        try:
            target = parsedate_to_datetime(stripped)
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        seconds = (target - datetime.now(UTC)).total_seconds()
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


@overload
def _request_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 600.0,
    response_json_required: bool = True,
    _boptest_status_response: Literal[False] = False,
) -> dict[str, Any]: ...


@overload
def _request_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 600.0,
    response_json_required: bool = True,
    _boptest_status_response: Literal[True],
) -> dict[str, Any] | str: ...


def _request_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 600.0,
    response_json_required: bool = True,
    _boptest_status_response: bool = False,
) -> dict[str, Any] | str:
    body = None if payload is None else _request_body(payload)
    request_headers = dict(headers or {})
    if payload is not None:
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            try:
                raw = response.read().decode("utf-8")
            except UnicodeDecodeError as error:
                raise TransportError(
                    f"endpoint returned non-UTF-8 text for {url}",
                    error_type="response_text_invalid",
                    provider_response_received=True,
                ) from error
            if response.status != 200:
                retryable = response.status in {429, 503}
                response_headers = getattr(response, "headers", None)
                retry_after = (
                    _retry_after_seconds(response_headers.get("Retry-After"))
                    if retryable and response_headers is not None
                    else None
                )
                raise TransportError(
                    f"HTTP {response.status} from {url}",
                    retryable=retryable,
                    error_type=f"http_{response.status}",
                    provider_response_received=True,
                    retry_after_seconds=retry_after,
                )
    except urllib.error.HTTPError as error:
        retryable = error.code in {429, 503}
        response_headers = getattr(error, "headers", None)
        raise TransportError(
            f"HTTP {error.code} from {url}",
            retryable=retryable,
            error_type=f"http_{error.code}",
            provider_response_received=True,
            retry_after_seconds=(
                _retry_after_seconds(response_headers.get("Retry-After"))
                if retryable and response_headers is not None
                else None
            ),
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead) as error:
        retryable, error_type = _connection_failure(error)
        raise TransportError(
            f"request failed for {url}: {error_type}",
            retryable=retryable,
            error_type=error_type,
        ) from error
    try:
        value = json.loads(raw) if raw else {}
    except json.JSONDecodeError as error:
        if not response_json_required:
            return {}
        raise TransportError(
            f"endpoint returned invalid JSON for {url}",
            error_type="response_json_invalid",
            provider_response_received=True,
        ) from error
    if _boptest_status_response and isinstance(value, str):
        if value in {"Running", "Queued"}:
            return value
        raise TransportError(
            f"BOPTEST status response is invalid for {url}",
            error_type="boptest_status_invalid",
            provider_response_received=True,
        )
    if not isinstance(value, dict):
        raise TransportError(
            f"endpoint returned a non-object for {url}",
            error_type="response_json_non_object",
            provider_response_received=True,
        )
    return value


LifecycleSink = Callable[[Mapping[str, Any]], None]


class BoptestHttpClient:
    def __init__(self, endpoint: str, *, queue_poll_seconds: float = 1.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.test_id: str | None = None
        self.testcase: str | None = None
        if not math.isfinite(queue_poll_seconds) or queue_poll_seconds <= 0:
            raise ValueError("BOPTEST queue poll interval must be positive")
        self.queue_poll_seconds = float(queue_poll_seconds)
        self._lifecycle_sink: LifecycleSink | None = None

    def set_lifecycle_sink(self, sink: LifecycleSink) -> None:
        self._lifecycle_sink = sink

    def _emit_lifecycle(self, *, event: str, status: str | None = None) -> None:
        if self._lifecycle_sink is None:
            return
        row: dict[str, Any] = {
            "phase": "physical_dispatch",
            "event": event,
            "dispatch_mode": "auto",
            "test_id": self.test_id,
            "testcase": self.testcase,
        }
        if status is not None:
            row["status"] = status
        self._lifecycle_sink(row)

    def status(self) -> str:
        if self.test_id is None:
            raise TransportError("BOPTEST status requested before select")
        response = _request_json(
            "GET",
            f"{self.endpoint}/status/{self.test_id}",
            _boptest_status_response=True,
        )
        status = response if isinstance(response, str) else response.get("payload")
        if not isinstance(status, str) or status not in {"Running", "Queued"}:
            raise TransportError(
                "BOPTEST status payload is invalid",
                error_type="boptest_status_invalid",
                provider_response_received=True,
            )
        return status

    def _wait_until_running(self) -> None:
        previous: str | None = None
        while True:
            status = self.status()
            if status != previous:
                self._emit_lifecycle(event="status_changed", status=status)
                previous = status
            if status == "Running":
                return
            time.sleep(self.queue_poll_seconds)

    def initialize(
        self, testcase: str, start_time_seconds: int, warmup_period_seconds: int
    ) -> dict[str, Any]:
        self.select_testcase(testcase)
        return self.initialize_selected(start_time_seconds, warmup_period_seconds)

    def select_testcase(self, testcase: str) -> str:
        """Select one worker and freeze its test id for this client."""
        if self.test_id is not None:
            raise TransportError("BOPTEST client already owns a live test id")
        selected = _request_json("POST", f"{self.endpoint}/testcases/{testcase}/select")
        test_id = selected.get("testid")
        if not isinstance(test_id, str) or not test_id:
            raise TransportError("BOPTEST select did not return a test id")
        self.test_id = test_id
        self.testcase = testcase
        self._emit_lifecycle(event="selected")
        self._wait_until_running()
        _request_json(
            "PUT",
            f"{self.endpoint}/scenario/{test_id}",
            payload={"electricity_price": "dynamic"},
        )
        _request_json("PUT", f"{self.endpoint}/step/{test_id}", payload={"step": 900})
        self._emit_lifecycle(event="configured", status="Running")
        return test_id

    def initialize_selected(
        self, start_time_seconds: int, warmup_period_seconds: int
    ) -> dict[str, Any]:
        """FMU-reset one selected test and repeat its complete BOPTEST warm-up."""

        if self.test_id is None:
            raise TransportError("BOPTEST initialize requested without a selected test id")
        self._wait_until_running()
        initialized = _request_json(
            "PUT",
            f"{self.endpoint}/initialize/{self.test_id}",
            payload={
                "start_time": start_time_seconds,
                "warmup_period": warmup_period_seconds,
            },
        )
        state = initialized.get("payload")
        if not isinstance(state, dict):
            raise TransportError("BOPTEST initialize payload is invalid")
        self._emit_lifecycle(event="initialized", status="Running")
        return state

    def forecast(
        self, points: Sequence[str], horizon_seconds: int, interval_seconds: int
    ) -> dict[str, list[float | None]]:
        if self.test_id is None:
            raise TransportError("BOPTEST forecast requested before initialize")
        response = _request_json(
            "PUT",
            f"{self.endpoint}/forecast/{self.test_id}",
            payload={
                "point_names": list(points),
                "horizon": horizon_seconds,
                "interval": interval_seconds,
            },
        )
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise TransportError("BOPTEST forecast payload is invalid")
        forecast: dict[str, list[float | None]] = {}
        for raw_point, raw_values in payload.items():
            point = str(raw_point)
            if not isinstance(raw_values, list):
                raise TransportError(f"BOPTEST forecast point {point} is not a list")
            values: list[float | None] = []
            for index, value in enumerate(raw_values):
                if value is None:
                    values.append(None)
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise TransportError(
                        f"BOPTEST forecast point {point} at index {index} is not a finite number"
                    )
                values.append(float(value))
            forecast[point] = values
        return forecast

    def advance(self, controls: Mapping[str, float]) -> dict[str, Any]:
        if self.test_id is None:
            raise TransportError("BOPTEST advance requested before initialize")
        response = _request_json(
            "POST", f"{self.endpoint}/advance/{self.test_id}", payload=controls
        )
        state = response.get("payload")
        if not isinstance(state, dict):
            raise TransportError("BOPTEST advance payload is invalid")
        return state

    def get_kpis(self) -> dict[str, Any]:
        """Read native BOPTEST KPIs before stopping the active test."""
        if self.test_id is None:
            raise TransportError("BOPTEST KPI requested before initialize")
        response = _request_json("GET", f"{self.endpoint}/kpi/{self.test_id}")
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise TransportError("BOPTEST KPI payload is invalid")
        return dict(payload)

    def stop(self) -> None:
        if self.test_id is None:
            raise TransportError("BOPTEST stop requested without a live test id")
        _request_json(
            "PUT",
            f"{self.endpoint}/stop/{self.test_id}",
            response_json_required=False,
        )
        self._emit_lifecycle(event="stopped")
        self.test_id = None

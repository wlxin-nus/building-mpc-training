"""Unchanged physical-only definitions extracted from the upstream module."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SPEC_VERSION = 2


PARAMETER_BOUNDS: dict[str, tuple[float, float, type[int] | type[float]]] = {
    "precool_lead_steps": (0, 4, int),
    "precool_residual_c": (-5.0, 5.0, float),
    "pmv_band_lo": (-0.5, 0.5, float),
    "pmv_band_hi": (-0.5, 0.5, float),
    "pmv_step_c": (0.0, 5.0, float),
}


BASE_CONDITION_FIELDS = ("occupied_now", "occupied_last", "precool_due", "last_pmv")


WEATHER_CONDITION_FIELDS = (
    "outdoor_temp_change_next_1h_c",
    "solar_irr_max_next_1h_w_m2",
    "solar_irr_mean_next_1h_w_m2",
)


RULE_OPERATORS = ("<", "<=", ">", ">=", "==", "!=")


RULE_ACTIONS = ("set_residual", "step_setpoint", "hold_setpoint")


_RULE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,23}$")


class ProgramError(ValueError):
    """Raised when a program or patch violates its deterministic contract."""

    def __init__(self, message: str, *, code: str = "program_invalid") -> None:
        super().__init__(message)
        self.code = code


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgramError(f"{name} must be numeric", code="invalid_number")
    number = float(value)
    if not math.isfinite(number):
        raise ProgramError(f"{name} must be finite", code="invalid_number")
    return number


def _resolved_value(program: Mapping[str, Any], value: Any) -> float:
    if isinstance(value, Mapping):
        if set(value) == {"neg_param"} and value["neg_param"] in PARAMETER_BOUNDS:
            return -float(program["params"][value["neg_param"]])
        if set(value) == {"param"} and value["param"] in PARAMETER_BOUNDS:
            return float(program["params"][value["param"]])
        raise ProgramError("value parameter reference is invalid", code="invalid_parameter")
    return _finite_number(value, "rule value")


def load_program(path: str | Path, zone: str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    program = {
        key: copy.deepcopy(raw[key])
        for key in ("spec_version", "precool_trigger", "params", "rules")
    }
    program["zone"] = zone
    program["provenance"] = {}
    validate_program(program)
    return program


def condition_fields(*, weather_enabled: bool = True) -> tuple[str, ...]:
    return BASE_CONDITION_FIELDS + (WEATHER_CONDITION_FIELDS if weather_enabled else ())


def validate_program(
    program: Mapping[str, Any], *, weather_enabled: bool = True
) -> Mapping[str, Any]:
    if program.get("spec_version") != SPEC_VERSION:
        raise ProgramError("unsupported program schema", code="schema_invalid")
    if program.get("precool_trigger") != "rolling":
        raise ProgramError("preconditioning trigger must be rolling", code="trigger_invalid")
    params = program.get("params")
    if not isinstance(params, Mapping) or set(params) != set(PARAMETER_BOUNDS):
        raise ProgramError(
            "program parameters do not match the canonical set", code="params_invalid"
        )
    for name, (lower, upper, kind) in PARAMETER_BOUNDS.items():
        value = _finite_number(params[name], name)
        if kind is int and value != int(value):
            raise ProgramError(f"{name} must be an integer", code="out_of_box")
        if not lower <= value <= upper:
            raise ProgramError(f"{name} is outside its bound", code="out_of_box")
    if not float(params["pmv_band_lo"]) < float(params["pmv_band_hi"]):
        raise ProgramError("PMV lower band must be below upper band", code="out_of_box")

    allowed_fields = set(condition_fields(weather_enabled=weather_enabled))
    rules = program.get("rules")
    if not isinstance(rules, list) or len(rules) > 8:
        raise ProgramError("rules must be a list with at most eight entries", code="rules_invalid")
    seen_ids: set[str] = set()
    seen_shapes: set[tuple[tuple[str, str, str], ...]] = set()
    for rule in rules:
        if not isinstance(rule, Mapping) or set(rule) != {"id", "when", "then"}:
            raise ProgramError("each rule must contain id, when, and then", code="rule_invalid")
        rule_id = rule["id"]
        if not isinstance(rule_id, str) or not _RULE_ID.fullmatch(rule_id) or rule_id in seen_ids:
            raise ProgramError("rule identifiers must be unique and short", code="rule_invalid")
        seen_ids.add(rule_id)
        conditions = rule["when"]
        if not isinstance(conditions, list) or not conditions:
            raise ProgramError("a rule needs at least one condition", code="rule_invalid")
        shape: list[tuple[str, str, str]] = []
        for condition in conditions:
            if not isinstance(condition, Mapping) or set(condition) != {"field", "op", "value"}:
                raise ProgramError(
                    "conditions must contain field, op, and value", code="rule_invalid"
                )
            if condition["field"] not in allowed_fields or condition["op"] not in RULE_OPERATORS:
                raise ProgramError(
                    "condition field or operator is unsupported", code="rule_invalid"
                )
            if condition["field"] in WEATHER_CONDITION_FIELDS and isinstance(
                condition["value"], Mapping
            ):
                raise ProgramError(
                    "weather conditions require numeric literals", code="rule_invalid"
                )
            _resolved_value(program, condition["value"])
            shape.append(
                (
                    str(condition["field"]),
                    str(condition["op"]),
                    json.dumps(condition["value"], sort_keys=True),
                )
            )
        shape_key = tuple(sorted(shape))
        if shape_key in seen_shapes:
            raise ProgramError("duplicate rule conditions", code="rule_invalid")
        seen_shapes.add(shape_key)
        action = rule["then"]
        if not isinstance(action, Mapping) or action.get("op") not in RULE_ACTIONS:
            raise ProgramError("rule action is unsupported", code="rule_invalid")
        if action["op"] == "hold_setpoint":
            if set(action) != {"op"}:
                raise ProgramError("hold_setpoint carries no value", code="rule_invalid")
        elif set(action) != {"op", "value"}:
            raise ProgramError("rule action requires a value", code="rule_invalid")
        else:
            value = _resolved_value(program, action["value"])
            if not -5.0 <= value <= 5.0:
                raise ProgramError("rule action is outside residual bounds", code="out_of_box")
    return program


def _state(program: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, float]:
    occupied = float(observation["current_occupancy"]) > 0
    lead = int(program["params"]["precool_lead_steps"])
    due = any(float(value) > 0 for value in (observation.get("occ_ahead") or [])[:lead])
    state = {
        "occupied_now": 1.0 if occupied else 0.0,
        "occupied_last": 1.0 if float(observation["last_occupancy"]) > 0 else 0.0,
        "precool_due": 1.0 if not occupied and due else 0.0,
        "last_pmv": float(observation["last_pmv"]),
    }
    used_weather = {
        condition["field"]
        for rule in program["rules"]
        for condition in rule["when"]
        if condition["field"] in WEATHER_CONDITION_FIELDS
    }
    for field in used_weather:
        if field not in observation:
            raise ProgramError(f"weather condition input is missing: {field}", code="missing_field")
        state[field] = _finite_number(observation[field], field)
    return state


def _matches(
    program: Mapping[str, Any], rule: Mapping[str, Any], state: Mapping[str, float]
) -> bool:
    for condition in rule["when"]:
        left = state[condition["field"]]
        right = _resolved_value(program, condition["value"])
        operation = condition["op"]
        matched = {
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
            "==": abs(left - right) < 1e-9,
            "!=": abs(left - right) >= 1e-9,
        }[operation]
        if not matched:
            return False
    return True


def run_program(program: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    state = _state(program, observation)
    base = 25.0 if state["occupied_now"] else 30.0
    residual = 0.0
    matched_rule: str | None = None
    for rule in program["rules"]:
        if not _matches(program, rule, state):
            continue
        action = rule["then"]["op"]
        value = (
            0.0 if action == "hold_setpoint" else _resolved_value(program, rule["then"]["value"])
        )
        if action == "set_residual":
            residual = value
        elif action == "step_setpoint":
            step_base = float(observation["last_setpoint"]) if state["occupied_last"] else base
            residual = step_base + value - base
        else:
            residual = float(observation["last_setpoint"]) - base
        matched_rule = str(rule["id"])
        break
    residual = max(-5.0, min(5.0, residual))
    setpoint = max(20.0, min(30.0, base + residual))
    rate_base = (
        float(observation["last_setpoint"])
        if state["occupied_now"] and state["occupied_last"]
        else base
    )
    if matched_rule == "precool":
        branch = "preconditioning"
    elif not state["occupied_now"]:
        branch = "unoccupied"
    else:
        branch_by_rule = {
            "pmv_raise": "occupied_pmv_raise",
            "pmv_lower": "occupied_pmv_lower",
        }
        branch = branch_by_rule.get(
            matched_rule if matched_rule is not None else "", "occupied_pmv_in_band"
        )
        if not state["occupied_last"]:
            branch = "occupancy_onset_" + branch
    return {
        "setpoint": setpoint,
        "base_setpoint": base,
        "base": rate_base,
        "residual": residual,
        "matched_rule": matched_rule,
        "branch": branch,
        "rules_fired": [matched_rule] if matched_rule is not None else [],
        "exempt_rate": not bool(state["occupied_now"]),
    }

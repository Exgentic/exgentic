# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import types
from typing import Any, Literal, Union, get_args, get_origin

from nicegui import ui

try:
    from pydantic.fields import PydanticUndefined
except Exception:  # pragma: no cover
    PydanticUndefined = None


def _is_optional(ann: Any) -> tuple[bool, Any]:
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        args = list(get_args(ann))
        if len(args) == 2 and type(None) in args:
            base = args[0] if args[1] is type(None) else args[1]
            return True, base
    return False, ann


def _build_pydantic_form(model_cls: type[Any], disabled: bool) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    fields = model_cls.model_fields
    schema = model_cls.model_json_schema()
    schema_props: dict[str, Any] = schema.get("properties", {})

    with ui.column().classes("w-full form-stack"):
        for name, form_field in fields.items():
            if name.startswith("_"):
                continue
            ann = form_field.annotation
            default = form_field.default
            if PydanticUndefined is not None and default is PydanticUndefined:
                default = None
            is_opt, base = _is_optional(ann)
            origin = get_origin(base)

            options = None
            prop = schema_props.get(name) or {}
            if isinstance(prop, dict):
                if "enum" in prop:
                    options = list(prop["enum"])
                elif "anyOf" in prop:
                    any_of = prop["anyOf"]
                    if isinstance(any_of, list):
                        for sub in any_of:
                            if isinstance(sub, dict) and "enum" in sub:
                                options = list(sub["enum"])
                                break
                elif "const" in prop:
                    options = [prop["const"]]
            if origin is Literal and options is None:
                options = list(get_args(base))

            if options:
                shown = ["", *options] if is_opt and default is None else options
                value = default if default in shown else (shown[0] if shown else None)
                control = ui.select(shown, value=value, label=name).props("dense")
                control.enabled = not disabled
                controls[name] = ("select", control, is_opt)
            elif base in (int, float):
                value = default if default is not None else 0
                control = ui.number(label=name, value=value)
                control.enabled = not disabled
                controls[name] = ("number", control, is_opt, base)
            elif base is bool:
                value = bool(default) if default is not None else False
                control = ui.checkbox(name, value=value)
                control.enabled = not disabled
                controls[name] = ("checkbox", control, is_opt)
            elif (get_origin(base) is dict) or (base is dict):
                init = json.dumps(default or {}, ensure_ascii=False, indent=2)
                control = ui.textarea(label=name, value=init).props("rows=4")
                control.enabled = not disabled
                controls[name] = ("json", control, is_opt, "dict")
            elif (get_origin(base) is list) or (base is list):
                init = json.dumps(default or [], ensure_ascii=False, indent=2)
                control = ui.textarea(label=name, value=init).props("rows=4")
                control.enabled = not disabled
                controls[name] = ("json", control, is_opt, "list")
            else:
                value = "" if default is None else str(default)
                control = ui.input(label=name, value=value)
                control.enabled = not disabled
                controls[name] = ("text", control, is_opt)

    return controls


def _build_agent_form(agent_cls: type[Any], disabled: bool) -> dict[str, Any]:
    return _build_pydantic_form(agent_cls, disabled)


def _collect_values(controls: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, data in controls.items():
        kind = data[0]
        control = data[1]
        is_opt = data[2] if len(data) > 2 else False
        raw = control.value
        if kind == "select":
            if is_opt and (raw is None or raw == ""):
                values[name] = None
            else:
                values[name] = raw
        elif kind == "number":
            base = data[3]
            if raw is None and is_opt:
                values[name] = None
            else:
                values[name] = base(raw) if raw is not None else 0
        elif kind == "checkbox":
            values[name] = bool(raw)
        elif kind == "json":
            if raw is None or raw == "":
                values[name] = {} if data[3] == "dict" else []
            else:
                try:
                    values[name] = json.loads(raw)
                except Exception:
                    values[name] = {} if data[3] == "dict" else []
        else:
            if is_opt and (raw is None or raw == ""):
                values[name] = None
            else:
                values[name] = raw
    return values

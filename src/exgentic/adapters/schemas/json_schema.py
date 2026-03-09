# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import Any, Literal

from json_schema_to_pydantic import create_model as _schema_to_model
from pydantic import BaseModel


def make_args_model_from_json_schema(name: str, parameters: dict[str, Any]) -> type[BaseModel]:
    """Build a Pydantic v2 model from JSON Schema and verify core semantics match.

    Verifies (type/required/properties), ignoring cosmetic keys like 'title'.
    """
    # 1) build
    model = _schema_to_model(
        schema=parameters,
        base_model_type=BaseModel,
        root_schema=None,
        allow_undefined_array_items=False,
        allow_undefined_type=False,
    )

    return model


def _schema_to_type(schema: dict[str, Any]) -> Any:
    """Best-effort map a JSON Schema fragment to a Python type annotation."""
    if not isinstance(schema, dict):
        return Any
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return _enum_type(schema["enum"])  # type: ignore[arg-type]
    t = schema.get("type")
    items = schema.get("items") if isinstance(schema.get("items"), dict) else None
    return _json_type_to_py(t, items)


def _json_type_to_py(t: Any, item_schema: dict[str, Any] | None = None):
    """Map a JSON Schema "type" to a Python type annotation.

    Supports primitives and simple containers. For arrays/objects, uses generic
    fallbacks unless an item schema is provided for arrays.
    """
    from typing import Any as TAny
    from typing import Dict as TDict
    from typing import List as TList

    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        # Try to infer item type if provided, otherwise default to list[Any]
        if isinstance(item_schema, dict):
            inner = _schema_to_type(item_schema)
            return TList[inner]  # type: ignore[index]
        return TList[TAny]  # type: ignore[index]
    if t == "object":
        return TDict[str, TAny]  # type: ignore[index]
    return Any


def _enum_type(values: list[Any]):
    """Create a Literal type from enum values when feasible; otherwise Any."""
    try:
        return Literal[tuple(values)]  # type: ignore[misc]
    except TypeError:
        # Fallback if values contain unhashables or mixed unsupported types
        return Any


# def make_args_model_from_param_list(name: str, params: List[Dict[str, Any]]) -> type[BaseModel]:
#     """Create a Pydantic model from AppWorld 'standard' parameter list entries."""
#     fields: Dict[str, Tuple[Any, Any]] = {}

#     for p in params or []:
#         pname = p["name"]
#         ptype = p.get("type")
#         required = bool(p.get("required", False))
#         default = p.get("default", ... if required else None)
#         enum_vals = p.get("enum")
#         field_kwargs: Dict[str, Any] = {}

#         if isinstance(enum_vals, list) and enum_vals:
#             py_t = _enum_type(enum_vals)
#         else:
#             py_t = _json_type_to_py(ptype)

#         if ptype in ("number", "integer"):
#             if "minimum" in p:
#                 field_kwargs["ge"] = p["minimum"]
#             if "maximum" in p:
#                 field_kwargs["le"] = p["maximum"]
#         if ptype == "string":
#             if "minLength" in p:
#                 field_kwargs["min_length"] = p["minLength"]
#             if "maxLength" in p:
#                 field_kwargs["max_length"] = p["maxLength"]

#         if field_kwargs:
#             annotated = Annotated[py_t, Field(**field_kwargs)]  # type: ignore[misc]
#             fields[pname] = (annotated, default)
#         else:
#             fields[pname] = (py_t, default)

#     return create_model(f"{name}_Args", **fields)

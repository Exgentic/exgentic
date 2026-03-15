# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import inspect
from typing import Any, Callable

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from ...core.actions import build_action
from ...core.types import ActionType, MultiObservation, SingleAction, SingleObservation


def action_type_to_function(
    action_type: "ActionType", internal_function: Callable[["SingleAction"], Any]
) -> Callable[..., Any]:
    def function(*args, **kwargs: Any) -> Any:
        all_kwargs = bind_arguments(cls=action_type.arguments, args=args, kwargs=kwargs)
        action = build_action(action_type, all_kwargs)
        observation = internal_function(action)
        if observation is None:
            return None
        if isinstance(observation, SingleObservation):
            return observation.result
        if isinstance(observation, MultiObservation):
            return [obs.result for obs in observation.observations]
        raise TypeError(f"Unexpected observation type: {type(observation).__name__}")

    function.__name__ = action_type.name.replace(".", "__")

    docstring_parts = [action_type.description]

    arguments_type = action_type.arguments
    if not isinstance(arguments_type, type) or not issubclass(arguments_type, BaseModel):
        raise TypeError(f"Action arguments must be a Pydantic BaseModel, got {arguments_type!r}")

    params = []
    annotations: dict[str, Any] = {}

    fields = arguments_type.model_fields
    if fields:
        docstring_parts.extend(["", "Args:"])
        for field_name, field_info in fields.items():
            anno = field_info.annotation or Any

            default = field_info.default
            required = default is PydanticUndefined
            if default is None:
                required = False

            param = inspect.Parameter(
                name=field_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=anno,
                default=(inspect._empty if required else default),
            )
            params.append(param)
            annotations[field_name] = anno

            # Description and Google-style formatting
            desc = field_info.description
            type_name = str(anno).replace("typing.", "")
            docstring_parts.append(f"    {field_name} ({type_name}): {desc}")

        function.__signature__ = inspect.Signature(parameters=params)
        function.__annotations__ = annotations
    else:
        function.__signature__ = inspect.Signature(parameters=[])
        function.__annotations__ = {}

    function.__annotations__["return"] = Any
    function.__doc__ = "\n".join(docstring_parts)

    return function


def bind_arguments(cls: type[BaseModel], args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bind positional args to Pydantic model fields by declaration order."""
    field_names = list(cls.model_fields.keys())
    if len(args) > len(field_names):
        raise TypeError(
            f"Too many positional arguments for {cls.__name__} "
            f"(expected at most {len(field_names)}, got {len(args)})"
        )

    positional_names = field_names[: len(args)]
    _duplicates = set(positional_names) & set(kwargs.keys())
    if _duplicates:
        dup_list = ", ".join(sorted(_duplicates))
        raise TypeError(f"Multiple values for argument(s): {dup_list}")

    bound = dict(zip(positional_names, args))
    bound.update(kwargs)
    return bound

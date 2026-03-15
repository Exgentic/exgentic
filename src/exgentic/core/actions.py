# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
from collections import Counter
from logging import Logger
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError

from ..observers.logging import get_disabled_logger
from .types import (
    Action,
    ActionType,
    MultiObservation,
    SingleAction,
    SingleObservation,
    ValidationReport,
)

ActionHandler = Callable[[SingleAction], Optional[Any]]


def _parse_arguments_payload(arguments: Any) -> Any:
    """Return arguments with best-effort JSON parsing for string inputs."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _validation_report_error(report: ValidationReport) -> Optional[str]:
    """Return a unified error string if the validation report marks the action invalid."""
    if not report.name_valid:
        return report.error or "Invalid action name"
    if not report.args_valid or not report.valid:
        return report.error or "Invalid arguments"
    return None


def build_action(action_type: ActionType, arguments: Any, *, action_id: Optional[str] = None) -> SingleAction:
    """Best-effort construction of a SingleAction with validity flag and optional ID."""
    parsed_args = _parse_arguments_payload(arguments)

    data: dict[str, Any] = {"name": action_type.name, "arguments": parsed_args}
    if action_id is not None:
        data["id"] = action_id

    report = ValidationReport()
    try:
        action = action_type.cls.model_validate(data)
    except ValidationError as exc:
        report.valid = False
        report.args_valid = False
        report.error = format_validation_errors(exc)
        report.details = {"errors": exc.errors()}
        args_cls = action_type.arguments
        if isinstance(parsed_args, dict) and isinstance(args_cls, type) and issubclass(args_cls, BaseModel):
            try:
                parsed_args = args_cls.model_validate(parsed_args)
                data["arguments"] = parsed_args
            except ValidationError:
                try:
                    parsed_args = args_cls.model_construct(**parsed_args)
                    data["arguments"] = parsed_args
                except Exception:
                    pass
        action = action_type.cls.model_construct(**data)

    # Attach validation metadata on the action instance
    try:
        action.validation = report  # type: ignore[attr-defined]
    except Exception:
        object.__setattr__(action, "validation", report)
    return action


def build_unknown_action(name: str, arguments: Any = None, *, action_id: Optional[str] = None) -> SingleAction:
    """Construct a best-effort unknown action marked invalid for name lookup paths."""
    parsed_args = _parse_arguments_payload(arguments)

    report = ValidationReport(
        valid=False,
        name_valid=False,
        args_valid=True,
        error="Unknown action",
        details={"reason": "unknown_action"},
    )
    payload: dict[str, Any] = {
        "name": name,
        "arguments": parsed_args if parsed_args is not None else {},
        "validation": report,
    }
    if action_id is not None:
        payload["id"] = action_id
    # model_construct to avoid BaseModel validation on arbitrary arguments types
    return SingleAction.model_construct(**payload)


def format_validation_errors(error: ValidationError) -> str:
    """Format pydantic validation errors into a compact human-readable string."""
    result = ""
    for err in error.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x is not Ellipsis) or "value"
        msg = err.get("msg", "Invalid value")
        input_value = err.get("input", None)
        input_type = type(input_value).__name__

        if err.get("type") == "missing":
            result += f"The field '{loc}' is required but was not provided."
        else:
            result += f"Field '{loc}': {msg} " f"(received {input_value!r} of type {input_type})."
    return result


def extract_argument(arguments: Any, field_name: str, default: Any = None) -> Any:
    """Best-effort extraction for a field from BaseModel/dict-like payloads."""
    if isinstance(arguments, BaseModel):
        return arguments.model_dump().get(field_name, default)
    if isinstance(arguments, dict):
        return arguments.get(field_name, default)
    return default


class ActionsHandler:
    """Central handler/registry for available actions and their handlers."""

    def __init__(
        self,
        logger: Optional[Logger] = None,
        *,
        warn_on_validation_error: bool = True,
        warn_on_unknown_action: bool = True,
        handle_validation_error: Optional[Callable[[SingleAction, str], Optional[SingleObservation]]] = None,
        handle_unknown_action: Optional[Callable[[SingleAction], Optional[SingleObservation]]] = None,
    ):
        if warn_on_unknown_action and handle_unknown_action is not None:
            raise ValueError("Cannot both warn and custom-handle unknown actions; set warn_on_unknown_action=False")
        if warn_on_validation_error and handle_validation_error is not None:
            raise ValueError("Cannot both warn and custom-handle validation errors; set warn_on_validation_error=False")
        self._logger = logger or get_disabled_logger()
        self._actions: dict[str, ActionType] = {}
        self._handlers: dict[str, ActionHandler] = {}
        self._warn_on_validation_error = warn_on_validation_error
        self._warn_on_unknown_action = warn_on_unknown_action
        self._handle_validation_error = handle_validation_error
        self._handle_unknown_action = handle_unknown_action or self._default_unknown_action
        self._stats: Counter[str] = Counter()

    # Registration ----------------------------------------------------------------
    def add_action(
        self,
        name: str,
        description: str,
        action_cls: type[SingleAction],
        handler: ActionHandler,
        *,
        is_finish: bool = False,
        is_message: bool = False,
        is_hidden: bool = False,
    ) -> ActionType:
        """Register a new action by specifying its parts; the ActionType is constructed internally."""
        action = ActionType(
            name=name,
            description=description,
            cls=action_cls,
            is_finish=is_finish,
            is_message=is_message,
            is_hidden=is_hidden,
        )
        self.add_action_type(action, handler)
        return action

    def add_action_type(self, action: ActionType, handler: ActionHandler) -> None:
        """Register an already-constructed ActionType."""
        if not isinstance(action, ActionType):
            raise ValueError("action must be an ActionType")
        self._store_action(action, handler)

    def add_actions(self, actions: list[ActionType], handler: ActionHandler) -> None:
        for action in actions:
            self.add_action_type(action, handler)

    # Accessors -------------------------------------------------------------------
    @property
    def actions(self) -> list[ActionType]:
        all_actions = list(self._actions.values())
        return list(filter(lambda action: not action.is_hidden, all_actions))

    def normalize(
        self,
        action: Optional[Action],
    ) -> Optional[SingleObservation | MultiObservation]:
        """Alias for execute() to emphasize validation/normalization use-cases."""
        return self.execute(action)

    # Execution -------------------------------------------------------------------
    def execute(
        self,
        action: Optional[Action],
    ) -> Optional[SingleObservation | MultiObservation]:
        """Execute user-supplied action(s) through registered handlers and return a merged observation."""
        if action is None:
            return None

        observations: list[SingleObservation] = []

        for single_action in action.to_action_list():
            outcome = self._execute_single(single_action)
            if outcome is not None:
                observations.append(outcome)

        if not observations:
            return None
        if len(observations) == 1:
            return observations[0]
        return MultiObservation(observations=observations)

    def _execute_single(self, action: SingleAction) -> Optional[SingleObservation]:
        handler = self._handlers.get(action.name)

        if handler is None:
            self._logger.error(f"Unknown action requested: {action.name}")
            self._record_error("unknown_action")
            return self._normalize_handler_result(self._handle_unknown_action(action), action)

        validation_error = self._validate_arguments(action)
        if validation_error:
            message = f"Validation Error in {action.name}: {validation_error}"
            self._logger.error(message)
            self._record_error("validation_error")
            if self._handle_validation_error is not None:
                return self._normalize_handler_result(
                    self._handle_validation_error(action, validation_error),
                    action,
                )
            if not self._warn_on_validation_error:
                return None
            return self._normalize_handler_result(message, action)

        try:
            raw_result = handler(action)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.exception(f"Action handler failed for {action.name}: {exc}")
            self._record_error("handler_exception")
            return self._normalize_handler_result(
                f"Action '{action.name}' failed: {exc}",
                action,
            )

        observation = self._normalize_handler_result(raw_result, action)
        return observation

    def _validate_arguments(self, action: SingleAction) -> Optional[str]:
        arguments = action.arguments
        expected_type = self._expected_arguments_type(action)

        report: Optional[ValidationReport] = action.validation
        if report:
            error = _validation_report_error(report)
            if error:
                return error
            return None

        # If arguments already a BaseModel, validate round-trip
        if isinstance(arguments, BaseModel):
            try:
                arguments.__class__.model_validate(arguments.model_dump())
            except ValidationError as e:
                return format_validation_errors(e)
            return None

        # If we know the expected type and got a dict, try to validate/construct
        if expected_type and issubclass(expected_type, BaseModel) and isinstance(arguments, dict):
            try:
                expected_type.model_validate(arguments)
            except ValidationError as e:
                return format_validation_errors(e)
            return None

        # Unknown/invalid argument shape
        if not isinstance(arguments, (BaseModel, dict)):
            msg = f"Invalid arguments type: {type(arguments).__name__}"
            if self._logger:
                self._logger.error(msg)
            self._record_error("invalid_arguments_type")
            return msg
        return None

    @staticmethod
    def _expected_arguments_type(action: SingleAction) -> Optional[type]:
        try:
            field = action.__class__.model_fields.get("arguments")
            if field and isinstance(field.annotation, type):
                return field.annotation  # type: ignore[return-value]
        except Exception:
            return None
        return None

    def _record_error(self, key: str) -> None:
        self._stats[key] += 1

    def get_errors_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _default_unknown_action(self, action: SingleAction) -> SingleObservation:
        if action.name == "message":
            text = "Error: Sending a message is not allowed. Please use only one of the available actions."
        else:
            text = f"Error: Unknown action - {action.name}"
        return SingleObservation(invoking_actions=[action], result=text)

    @staticmethod
    def _normalize_handler_result(raw_result: Any, action: SingleAction) -> Optional[SingleObservation]:
        if raw_result is None:
            return None
        if isinstance(raw_result, SingleObservation):
            if not raw_result.invoking_actions:
                raw_result.invoking_actions = [action]
            return raw_result
        return SingleObservation(invoking_actions=[action], result=raw_result)

    def _store_action(self, action: ActionType, handler: ActionHandler) -> None:
        self._validate_action_type(action)
        self._actions[action.name] = action
        self._handlers[action.name] = handler

    @staticmethod
    def _validate_action_type(action: ActionType) -> None:
        args_type = action.arguments
        if not isinstance(args_type, type) or not issubclass(args_type, BaseModel):
            raise ValueError(
                "Action arguments must be a Pydantic BaseModel. "
                f"Action '{action.name}' has arguments type {args_type!r}. "
                "Ensure the action's 'arguments' annotation resolves to a BaseModel."
            )

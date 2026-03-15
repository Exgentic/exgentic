# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from exgentic.core.actions import ActionsHandler, build_action, build_unknown_action
from exgentic.core.types import ActionType, SingleAction, SingleObservation
from pydantic import BaseModel


class Args(BaseModel):
    x: int


class MyAction(SingleAction):
    arguments: Args


def make_action_type(name: str = "do") -> ActionType:
    return ActionType(name=name, description="desc", cls=MyAction)


def test_build_action_valid():
    action_type = make_action_type()
    action = build_action(action_type, {"x": 1})

    assert action.validation.valid
    assert action.arguments.x == 1


def test_build_action_invalid_args_sets_report():
    action_type = make_action_type()
    action = build_action(action_type, {"x": "oops"})

    assert not action.validation.valid
    assert not action.validation.args_valid
    assert action.validation.error


def test_execute_unknown_action_returns_warning_and_stats():
    registry = ActionsHandler()
    action = MyAction(name="unknown", arguments=Args(x=1))

    observation = registry.execute(action)

    assert isinstance(observation, SingleObservation)
    assert "Unknown action" in str(observation.result)
    assert registry.get_errors_stats().get("unknown_action") == 1


def test_build_unknown_action_sets_validation():
    action = build_unknown_action("unknown_tool", {"foo": "bar"})
    assert not action.validation.valid
    assert not action.validation.name_valid
    assert action.validation.error == "Unknown action"


def test_build_unknown_action_parses_json_string_arguments():
    action = build_unknown_action("unknown_tool", '{"foo": 1}')
    assert isinstance(action.arguments, dict)
    assert action.arguments["foo"] == 1


def test_execute_validation_error_warns_and_counts():
    registry = ActionsHandler(warn_on_validation_error=True)
    action_type = make_action_type()
    registry.add_action_type(action_type, handler=lambda a: {"ok": a.arguments.x})

    action = build_action(action_type, {"x": "bad"})
    observation = registry.execute(action)

    assert isinstance(observation, SingleObservation)
    assert "Validation Error in do:" in str(observation.result)
    assert registry.get_errors_stats().get("validation_error") == 1


def test_execute_validation_error_custom_handler():
    action_type = make_action_type()
    registry = ActionsHandler(
        warn_on_validation_error=False,
        handle_validation_error=lambda action, msg: SingleObservation(
            invoking_actions=[action], result=f"handled:{msg}"
        ),
    )
    registry.add_action_type(action_type, handler=lambda a: {"ok": a.arguments.x})

    action = build_action(action_type, {"x": "bad"})
    observation = registry.execute(action)

    assert isinstance(observation, SingleObservation)
    assert str(observation.result).startswith("handled:")
    assert registry.get_errors_stats().get("validation_error") == 1


def test_handler_exception_wrapped_as_observation():
    registry = ActionsHandler()
    action_type = make_action_type("boom")

    def boom_handler(_action: SingleAction):
        raise RuntimeError("fail")

    registry.add_action_type(action_type, handler=boom_handler)
    action = build_action(action_type, {"x": 1})

    observation = registry.execute(action)

    assert isinstance(observation, SingleObservation)
    assert "Action 'boom' failed" in str(observation.result)
    assert registry.get_errors_stats().get("handler_exception") == 1


def test_unknown_message_action_has_friendly_message():
    registry = ActionsHandler()
    action = SingleAction.model_construct(name="message", arguments={})

    observation = registry.execute(action)

    assert isinstance(observation, SingleObservation)
    assert "Sending a message is not allowed" in str(observation.result)

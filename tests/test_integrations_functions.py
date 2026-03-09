# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import pytest
from exgentic.adapters.actions.functions import action_type_to_function, bind_arguments
from exgentic.core.types import Action, ActionType, SingleAction, SingleObservation
from pydantic import BaseModel


def test_bind():
    class Args(BaseModel):
        arg1: int
        arg2: int
        arg3: int
        arg4: int

    # normal flow
    expected = {"arg1": 1, "arg2": 2, "arg3": 3, "arg4": 4}
    res = bind_arguments(cls=Args, args=[1, 2], kwargs={"arg3": 3, "arg4": 4})
    assert res == expected

    # missing positional arguments
    expected = {"arg1": 1, "arg3": 3, "arg4": 4}
    res = bind_arguments(cls=Args, args=[1], kwargs={"arg3": 3, "arg4": 4})
    assert res == expected

    # no positional arguments
    expected = {"arg3": 3, "arg4": 4}
    res = bind_arguments(cls=Args, args=[], kwargs={"arg3": 3, "arg4": 4})
    assert res == expected

    # missing keyword arguments
    expected = {"arg1": 1, "arg2": 2, "arg4": 4}
    res = bind_arguments(cls=Args, args=[1, 2], kwargs={"arg4": 4})
    assert res == expected

    # no keyword arguments
    expected = {"arg1": 1, "arg2": 2}
    res = bind_arguments(cls=Args, args=[1, 2], kwargs={})
    assert res == expected

    # duplicates
    with pytest.raises(TypeError):
        bind_arguments(cls=Args, args=[1, 2], kwargs={"arg2": 2, "arg3": 3})

    # too many args
    with pytest.raises(TypeError):
        bind_arguments(cls=Args, args=[1, 2, 3, 4, 5], kwargs={})


def test_action_type_to_function():
    def internal_function(action: Action):
        return SingleObservation(result=action)

    class MyArgs(BaseModel):
        arg1: int
        arg2: int
        arg3: int = 0
        arg4: int

    class MyAction(SingleAction):
        name: str = "my_action"
        arguments: MyArgs

    action_type = ActionType(name="my_action", description="my description", cls=MyAction)

    function = action_type_to_function(action_type, internal_function)

    expected_arguments = MyArgs(arg1=1, arg2=2, arg3=3, arg4=4)
    action = function(1, 2, 3, 4)
    assert action.arguments == expected_arguments

    expected_arguments = MyArgs(arg1=1, arg2=2, arg3=3, arg4=4)
    action = function(1, 2, 3, arg4=4)
    assert action.arguments == expected_arguments

    expected_arguments = MyArgs(arg1=1, arg2=2, arg4=4)
    action = function(1, 2, arg4=4)
    assert action.arguments == expected_arguments

    """invalid cases allows only by model_construct"""
    # # missing required argument
    # with pytest.raises(ValidationError):
    #     action = function(1,arg4=4)

    # # non existing argname
    # with pytest.raises(ValidationError):
    #     action = function(1,2,3,arg4=4,arg5=5)

    # with pytest.raises(ValidationError):
    #     action = function(1,2,3, arg5=5)

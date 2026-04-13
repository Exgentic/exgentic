# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Mixin for version-safe pickling of pydantic models.

When pydantic models cross process boundaries via cloudpickle (e.g. the
venv or process runner), the default pickle behaviour serializes internal
pydantic state that is tied to the pydantic version.  If the host and
runner venvs have different pydantic versions, deserialization fails.

This mixin overrides ``__reduce__`` to serialize via ``model_dump()``
and reconstruct via ``model_validate()``, making the pickle payload a
plain dict that any pydantic version can consume.
"""

from __future__ import annotations

from typing import Any


def _restore(cls: type, data: dict[str, Any]) -> Any:
    return cls.model_validate(data)


class PickleSafe:
    """Mixin for pydantic ``BaseModel`` subclasses.

    Place **before** ``BaseModel`` in the class bases so that the
    ``__reduce__`` override takes precedence::

        class MyModel(PickleSafe, BaseModel):
            ...
    """

    def __reduce__(self) -> tuple:
        return (_restore, (type(self), self.model_dump()))

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

from framework.integrations.litellm.cache import (
    CustomCache,
    strip_date_time_from_text,
)


def test_strip_date_time_from_text_removes_date_time() -> None:
    text = "Schedule 2024-05-01 at 09:30 AM for review."
    cleaned = strip_date_time_from_text(text)
    assert "2024-05-01" not in cleaned
    assert "09:30" not in cleaned


def test_cache_key_ignores_date_time_when_enabled() -> None:
    cache = CustomCache(type="local", delete_time_from_messages=True)
    key_one = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Today is 2024-05-01 10:30."}],
    )
    key_two = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Today is 2025-06-02 22:15."}],
    )
    assert key_one == key_two


def test_cache_key_keeps_date_time_when_disabled() -> None:
    cache = CustomCache(type="local", delete_time_from_messages=False)
    key_one = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Today is 2024-05-01 10:30."}],
    )
    key_two = cache.get_cache_key(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Today is 2025-06-02 22:15."}],
    )
    assert key_one != key_two


def test_strip_date_does_not_remove_adjacent_numbers() -> None:
    r"""Regression test: numbers following a date should not be stripped.

    The original regex pattern `(?:,?\s+\d{2,4})?` was too permissive,
    matching 2-4 digit numbers after a space as if they were years.

    Example 1 - Broken datetime parsing:
      Input: "Jan 20 15:36"
      Bug: Date regex matches "Jan 20 15" (treats hour "15" as a year)
      Result: Leaves ":36" orphaned, time regex fails (no word boundary before ":")

    Example 2 - Lost data:
      Input: "May 15 30 items"
      Bug: Date regex matches "May 15 30" (treats "30" as a year)
      Result: Loses the number "30" entirely

    The fix uses `(?:,\s*\d{2}|\s+\d{4})?` which requires:
    - Comma followed by exactly 2-digit year (e.g., "May 15,23")
    - Space followed by exactly 4-digit year (e.g., "May 15 2023")
    """
    # Example 1: Datetime parsing - hour mistaken for year
    # Original bug: "Jan 20 15:36" -> matches "Jan 20 15", orphans ":36"
    text = "Event at Jan 20 15:36 in the main hall"
    cleaned = strip_date_time_from_text(text)
    assert "Jan 20" not in cleaned
    assert "15:36" not in cleaned  # time should be stripped properly
    assert ":36" not in cleaned  # no orphaned time fragment
    assert cleaned == "Event at in the main hall"

    # Example 2: Adjacent numbers - count mistaken for year
    # Original bug: "May 15 30" -> stripped entirely, losing "30"
    text2 = "May 15 30 items in stock"
    cleaned2 = strip_date_time_from_text(text2)
    assert "May 15" not in cleaned2
    assert "30" in cleaned2
    assert cleaned2 == "30 items in stock"

    # Example 3: 3-digit numbers are also not years
    # Original bug: "May 15 300" -> stripped entirely
    text3 = "May 15 300 users joined"
    cleaned3 = strip_date_time_from_text(text3)
    assert "300" in cleaned3
    assert cleaned3 == "300 users joined"

    # Valid year formats should still be stripped correctly
    assert strip_date_time_from_text("May 15 2023 was great") == "was great"
    assert strip_date_time_from_text("May 15, 23 ended") == "ended"

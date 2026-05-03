# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from exgentic.observers.handlers.session_timings import SessionLLMTimings, get_session_llm_timings


class TestSessionLLMTimings:
    def test_record_and_pop_returns_durations_in_order(self):
        t = SessionLLMTimings()
        t.record("sess-1", 0.5)
        t.record("sess-1", 1.25)
        t.record("sess-1", 0.75)
        assert t.pop("sess-1") == [0.5, 1.25, 0.75]

    def test_pop_unknown_session_returns_empty_list(self):
        assert SessionLLMTimings().pop("missing") == []

    def test_pop_clears_durations(self):
        t = SessionLLMTimings()
        t.record("sess-1", 0.1)
        t.pop("sess-1")
        assert t.pop("sess-1") == []

    def test_record_isolates_sessions(self):
        t = SessionLLMTimings()
        t.record("sess-A", 0.1)
        t.record("sess-B", 0.2)
        assert t.pop("sess-A") == [0.1]
        assert t.pop("sess-B") == [0.2]

    def test_negative_duration_silently_ignored(self):
        t = SessionLLMTimings()
        t.record("sess-1", -1.0)
        assert t.pop("sess-1") == []

    def test_empty_session_id_silently_ignored(self):
        t = SessionLLMTimings()
        t.record("", 1.0)
        assert t.pop("") == []

    def test_module_singleton_returns_same_instance(self):
        assert get_session_llm_timings() is get_session_llm_timings()

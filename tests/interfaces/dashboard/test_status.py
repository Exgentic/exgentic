# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.


from exgentic.interfaces.dashboard.views.status import (
    _status_counts_from_sessions,
    _status_from_outcome,
)


class TestStatusFromOutcome:
    def test_success(self):
        assert _status_from_outcome(True, True) == "success"

    def test_success_overrides_error_source(self):
        assert _status_from_outcome(True, True, error_source="agent") == "success"

    def test_unsuccessful(self):
        assert _status_from_outcome(False, True) == "unsuccessful"

    def test_unfinished(self):
        assert _status_from_outcome(False, False) == "unfinished"

    def test_agent_error(self):
        assert _status_from_outcome(False, None, error_source="agent") == "agent error"

    def test_benchmark_error(self):
        assert _status_from_outcome(False, None, error_source="benchmark") == "benchmark error"

    def test_cancelled(self):
        assert _status_from_outcome(False, None, error_source="cancelled") == "cancelled"

    def test_generic_error(self):
        assert _status_from_outcome(False, None) == "error"

    def test_none_success_none_finished(self):
        assert _status_from_outcome(None, None) == "error"

    def test_finished_true_but_not_success(self):
        assert _status_from_outcome(None, True) == "unsuccessful"


class TestStatusCountsFromSessions:
    def test_empty_sessions(self):
        assert _status_counts_from_sessions({}) == {}

    def test_single_status(self):
        sessions = {"s1": {"status": "success"}, "s2": {"status": "success"}}
        assert _status_counts_from_sessions(sessions) == {"success": 2}

    def test_mixed_statuses(self):
        sessions = {
            "s1": {"status": "success"},
            "s2": {"status": "error"},
            "s3": {"status": "running"},
            "s4": {"status": "success"},
        }
        counts = _status_counts_from_sessions(sessions)
        assert counts == {"success": 2, "error": 1, "running": 1}

    def test_missing_status_defaults_to_error(self):
        sessions = {"s1": {"steps": 5}}
        assert _status_counts_from_sessions(sessions) == {"error": 1}

    def test_all_status_types(self):
        sessions = {
            "s1": {"status": "success"},
            "s2": {"status": "unsuccessful"},
            "s3": {"status": "unfinished"},
            "s4": {"status": "agent error"},
            "s5": {"status": "benchmark error"},
            "s6": {"status": "cancelled"},
            "s7": {"status": "error"},
            "s8": {"status": "running"},
        }
        counts = _status_counts_from_sessions(sessions)
        assert len(counts) == 8
        assert all(v == 1 for v in counts.values())

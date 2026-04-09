# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json

import pytest
from exgentic.interfaces.dashboard.views.data import (
    _build_history_sessions,
    _build_overview_secondary_metrics,
    _build_session_rows,
    _load_text_file,
    _load_trajectory_events,
    _resolve_planned_sessions,
    _resolve_run_meta,
    _resolve_total_workers,
    _short_model_name,
    load_leaderboard_data,
    load_run_results,
    load_session_file,
)


class TestShortModelName:
    def test_with_slash(self):
        assert _short_model_name("anthropic/claude-3") == "claude-3"

    def test_without_slash(self):
        assert _short_model_name("gpt-4") == "gpt-4"

    def test_non_string(self):
        assert _short_model_name(None) == "None"


class TestBuildOverviewSecondaryMetrics:
    def test_from_results(self):
        results = {
            "average_score": 0.85,
            "average_steps": 10,
            "average_agent_cost": 0.5,
            "average_benchmark_cost": 0.1,
            "total_run_cost": 3.0,
            "percent_successful": 0.8,
            "percent_finished": 0.9,
        }
        metrics = _build_overview_secondary_metrics({}, results)
        labels = [m[0] for m in metrics]
        assert "Avg Score" in labels
        assert "Total Run Cost" in labels
        assert "Success Rate" in labels
        assert "Finished Rate" in labels
        values = dict(metrics)
        assert values["Avg Score"] == 0.85
        assert values["Total Run Cost"] == 3.0
        assert values["Success Rate"] == 0.8
        assert values["Finished Rate"] == 0.9

    def test_from_sessions_fallback(self):
        sessions = {
            "s1": {"status": "success", "steps": 5, "score": 0.8, "agent_cost": 1.0, "benchmark_cost": 0.2},
            "s2": {"status": "error", "steps": 3, "score": 0.6, "agent_cost": 0.5, "benchmark_cost": 0.1},
        }
        metrics = _build_overview_secondary_metrics(sessions, None)
        values = dict(metrics)
        assert values["Avg Score"] == pytest.approx(0.7)
        assert values["Avg Steps"] == pytest.approx(4.0)
        assert values["Total Run Cost"] == pytest.approx(1.8)
        assert values["Success Rate"] == pytest.approx(0.5)
        assert values["Finished Rate"] == pytest.approx(0.5)

    def test_empty_sessions_no_results(self):
        metrics = _build_overview_secondary_metrics({}, None)
        values = dict(metrics)
        assert values["Avg Score"] is None
        assert values["Total Run Cost"] is None
        assert values["Success Rate"] is None
        assert values["Finished Rate"] is None

    def test_sessions_with_no_cost(self):
        sessions = {"s1": {"status": "success", "steps": 2}}
        metrics = _build_overview_secondary_metrics(sessions, None)
        values = dict(metrics)
        assert values["Total Run Cost"] is None

    def test_rates_from_completed_sessions(self):
        sessions = {
            "s1": {"status": "success"},
            "s2": {"status": "unsuccessful"},
            "s3": {"status": "error"},
            "s4": {"status": "running"},
        }
        metrics = _build_overview_secondary_metrics(sessions, None)
        values = dict(metrics)
        # running sessions are excluded from rate calculations
        # 3 completed: 1 success, 1 unsuccessful, 1 error
        assert values["Success Rate"] == pytest.approx(1 / 3)
        assert values["Finished Rate"] == pytest.approx(2 / 3)  # success + unsuccessful


class TestBuildSessionRows:
    def test_basic(self):
        sessions = {
            "s1": {"status": "success", "steps": 5, "score": 1.0},
            "s2": {"status": "running", "steps": 2, "score": None},
        }
        rows = _build_session_rows(sessions)
        assert len(rows) == 2
        row_map = {r["session"]: r for r in rows}
        assert row_map["s1"]["status"] == "success"
        assert row_map["s2"]["steps"] == 2

    def test_empty(self):
        assert _build_session_rows({}) == []

    def test_missing_fields(self):
        sessions = {"s1": {}}
        rows = _build_session_rows(sessions)
        assert rows[0]["status"] == ""
        assert rows[0]["steps"] == 0
        assert rows[0]["score"] is None


class TestResolveRunMeta:
    def test_from_results(self):
        results = {
            "benchmark_name": "bench1",
            "agent_name": "agent1",
            "model_names": ["anthropic/claude-3", "openai/gpt-4"],
        }
        meta = _resolve_run_meta(results, None)
        assert meta["benchmark"] == "bench1"
        assert meta["agent"] == "agent1"
        assert meta["models"] == ["claude-3", "gpt-4"]

    def test_from_config(self):
        config = {
            "benchmark": {"slug_name": "bench-slug"},
            "agent": {"slug_name": "agent-slug", "model_names": ["m1"]},
        }
        meta = _resolve_run_meta(None, config)
        assert meta["benchmark"] == "bench-slug"
        assert meta["agent"] == "agent-slug"
        assert meta["models"] == ["m1"]

    def test_fallback(self):
        meta = _resolve_run_meta(None, None, fallback_benchmark="fb", fallback_agent="fa")
        assert meta["benchmark"] == "fb"
        assert meta["agent"] == "fa"

    def test_no_data(self):
        meta = _resolve_run_meta(None, None)
        assert meta["benchmark"] == "-"
        assert meta["agent"] == "-"
        assert meta["models"] == []

    def test_model_dedup(self):
        results = {"model_names": ["a", "b", "a"]}
        meta = _resolve_run_meta(results, None)
        assert meta["models"] == ["a", "b"]

    def test_string_benchmark_in_config(self):
        config = {"benchmark": "simple-name", "agent": "agent-name"}
        meta = _resolve_run_meta(None, config)
        assert meta["benchmark"] == "simple-name"
        assert meta["agent"] == "agent-name"


class TestResolvePlannedSessions:
    def test_from_results(self):
        assert _resolve_planned_sessions({"planned_sessions": 10}, None, None) == 10

    def test_from_config_planned(self):
        assert _resolve_planned_sessions(None, {"planned_sessions": 5}, None) == 5

    def test_from_config_num_tasks(self):
        assert _resolve_planned_sessions(None, {"num_tasks": 8}, None) == 8

    def test_from_config_run_block(self):
        config = {"run": {"num_tasks": 3}}
        assert _resolve_planned_sessions(None, config, None) == 3

    def test_fallback(self):
        assert _resolve_planned_sessions(None, None, 42) == 42

    def test_none(self):
        assert _resolve_planned_sessions(None, None, None) is None


class TestResolveTotalWorkers:
    def test_from_results(self):
        assert _resolve_total_workers({"max_workers": 4}, None, None) == 4

    def test_from_config(self):
        assert _resolve_total_workers(None, {"max_workers": 2}, None) == 2

    def test_from_config_run_block(self):
        config = {"run": {"max_workers": 6}}
        assert _resolve_total_workers(None, config, None) == 6

    def test_fallback(self):
        assert _resolve_total_workers(None, None, 1) == 1


class TestLoadRunResults:
    def test_valid_file(self, tmp_path):
        f = tmp_path / "results.json"
        f.write_text(json.dumps({"score": 0.9}))
        result = load_run_results(str(f))
        assert result == {"score": 0.9}

    def test_missing_file(self, tmp_path):
        assert load_run_results(str(tmp_path / "nope.json")) is None

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json")
        assert load_run_results(str(f)) is None


class TestLoadSessionFile:
    def test_json_file(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"key": "val"}))
        assert load_session_file(str(f), "json") == {"key": "val"}

    def test_log_file(self, tmp_path):
        f = tmp_path / "output.log"
        lines = ["line1\n", "line2\n"]
        f.write_text("".join(lines))
        result = load_session_file(str(f), "log")
        assert "line1" in result
        assert "line2" in result

    def test_log_limits_to_200_lines(self, tmp_path):
        f = tmp_path / "big.log"
        f.write_text("".join(f"line{i}\n" for i in range(300)))
        result = load_session_file(str(f), "log")
        assert "line100" in result
        assert "line299" in result

    def test_missing_file(self):
        assert load_session_file("/nonexistent/path.json", "json") is None


class TestLoadTrajectoryEvents:
    def test_valid_jsonl(self, tmp_path):
        f = tmp_path / "trajectory.jsonl"
        lines = [json.dumps({"event": "action", "step": i}) for i in range(3)]
        f.write_text("\n".join(lines))
        events = _load_trajectory_events(f)
        assert len(events) == 3
        assert events[0]["step"] == 0

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert _load_trajectory_events(f) == []

    def test_missing_file(self, tmp_path):
        assert _load_trajectory_events(tmp_path / "nope.jsonl") == []

    def test_skip_bad_lines(self, tmp_path):
        f = tmp_path / "mixed.jsonl"
        f.write_text('{"ok": true}\nnot json\n{"ok": false}\n')
        events = _load_trajectory_events(f)
        assert len(events) == 2


class TestLoadLeaderboardData:
    def test_loads_results(self, tmp_path):
        run_dir = tmp_path / "run-001"
        run_dir.mkdir()
        results = {
            "agent_name": "MyAgent",
            "model_name": "gpt-4",
            "benchmark_name": "SWE",
            "subset_name": "lite",
            "total_sessions": 50,
            "average_score": 0.75,
            "total_run_cost": 10.0,
            "average_agent_cost": 0.2,
        }
        (run_dir / "results.json").write_text(json.dumps(results))
        rows = load_leaderboard_data(str(tmp_path))
        assert len(rows) == 1
        assert rows[0]["Agent"] == "MyAgent"
        assert rows[0]["Num Tasks"] == 50

    def test_skips_invalid(self, tmp_path):
        run_dir = tmp_path / "bad-run"
        run_dir.mkdir()
        (run_dir / "results.json").write_text("broken")
        assert load_leaderboard_data(str(tmp_path)) == []

    def test_missing_dir(self, tmp_path):
        assert load_leaderboard_data(str(tmp_path / "nope")) == []

    def test_multiple_runs(self, tmp_path):
        for i in range(3):
            d = tmp_path / f"run-{i}"
            d.mkdir()
            (d / "results.json").write_text(json.dumps({"agent_name": f"agent{i}"}))
        rows = load_leaderboard_data(str(tmp_path))
        assert len(rows) == 3


class TestBuildHistorySessions:
    def test_from_session_results(self, tmp_path):
        results = {
            "session_results": [
                {
                    "session_id": "s1",
                    "success": True,
                    "is_finished": True,
                    "steps": 5,
                    "score": 1.0,
                    "execution_time": 30.0,
                    "agent_cost": 0.5,
                    "benchmark_cost": 0.1,
                    "details": {"session_metadata": {}},
                },
            ]
        }
        sessions = _build_history_sessions("run-1", results, output_dir=str(tmp_path))
        assert "s1" in sessions
        assert sessions["s1"]["status"] == "success"
        assert sessions["s1"]["steps"] == 5

    def test_fallback_to_filesystem(self, tmp_path):
        # Set up directory structure matching RunPaths
        run_dir = tmp_path / "run-1"
        sessions_dir = run_dir / "sessions"
        sess_dir = sessions_dir / "sess-a"
        sess_dir.mkdir(parents=True)
        (sess_dir / "results.json").write_text(
            json.dumps(
                {
                    "success": False,
                    "is_finished": True,
                    "steps": 3,
                    "score": 0.0,
                    "details": {"session_metadata": {}},
                }
            )
        )
        sessions = _build_history_sessions("run-1", None, output_dir=str(tmp_path))
        assert "sess-a" in sessions
        assert sessions["sess-a"]["status"] == "unsuccessful"

    def test_empty_results(self, tmp_path):
        sessions = _build_history_sessions("run-1", {}, output_dir=str(tmp_path))
        assert sessions == {}


class TestLoadTextFile:
    def test_json_file(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"a": 1}))
        result = _load_text_file(f)
        assert '"a": 1' in result

    def test_log_file(self, tmp_path):
        f = tmp_path / "out.log"
        f.write_text("log content here")
        result = _load_text_file(f)
        assert "log content here" in result

    def test_txt_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("some text")
        result = _load_text_file(f)
        assert "some text" in result

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")
        assert _load_text_file(f) is None

    def test_missing_file(self, tmp_path):
        assert _load_text_file(tmp_path / "nope.json") is None

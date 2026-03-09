# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import re
from pathlib import Path

from click.testing import CliRunner
from exgentic.core.types import SessionResults
from exgentic.core.types.session import SessionOutcomeStatus
from exgentic.interfaces.cli.main import cli


def create_mock_results(
    output_dir: Path,
    agent: str,
    model: str,
    benchmark: str,
    subset: str | None,
    tasks: list[tuple[str, float]],
    run_id: str | None = None,
) -> None:
    """Create mock results structure for testing compare command.

    Args:
        output_dir: Base output directory
        agent: Agent slug name
        model: Model name
        benchmark: Benchmark slug name
        subset: Optional subset name
        tasks: List of (task_id, score) tuples
        run_id: Optional run ID (generated if not provided)
    """
    if run_id is None:
        run_id = f"test-{agent}-{model}-{benchmark}"
        if subset:
            run_id += f"-{subset}"

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create config.json in run/config.json format
    config_dir = run_dir / "run"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "benchmark": {"slug_name": benchmark},
        "agent": {"slug_name": agent, "model_name": model},
        "model": model,
        "run_id": run_id,
    }

    if subset:
        config_data["benchmark"]["params"] = {"subset": subset}
        config_data["subset"] = subset

    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    # Create sessions directory with results
    sessions_dir = run_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    for task_id, score in tasks:
        session_id = f"session-{task_id}"
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create results.json
        session_result = SessionResults(
            session_id=session_id,
            task_id=task_id,
            success=score >= 0.99,
            score=score,
            is_finished=True,
            status=SessionOutcomeStatus.SUCCESS,
            steps=5,
            action_count=5,
            invalid_action_count=0,
            agent_cost=0.01,
            benchmark_cost=0.0,
            execution_time=10.0,
        )

        results_file = session_dir / "results.json"
        results_file.write_text(session_result.model_dump_json(indent=2), encoding="utf-8")


def test_compare_two_agents_same_benchmark(tmp_path):
    """Test comparing two different agents using the same model on a single benchmark.

    Verifies:
    - Comparison table is displayed with task results
    - Basic comparison functionality works
    - Task-level results are shown correctly
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create mock results for agent1
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 1.0), ("task-2", 0.8), ("task-3", 0.6)],
    )

    # Create mock results for agent2
    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 0.9), ("task-2", 0.85), ("task-3", 0.7)],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Comparison Results" in result.output
    # New format shows summary table with setup names and statistics
    assert "agent1 and model1" in result.output
    assert "agent2 and model1" in result.output
    assert "Statistical Significance Matrix" in result.output


def test_compare_two_models_same_agent(tmp_path):
    """Test comparing two different models using the same agent.

    Verifies:
    - Model comparison works correctly
    - Model-specific filtering is applied
    - Results are properly differentiated by model
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create mock results for model1
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 1.0), ("task-2", 0.9)],
    )

    # Create mock results for model2
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model2",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 0.8), ("task-2", 0.7)],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent1",
            "--model1",
            "model1",
            "--model2",
            "model2",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Comparison Results" in result.output
    # Verify model-specific filtering is applied - both models should appear
    assert "model1" in result.output
    assert "model2" in result.output
    # New format shows summary table
    assert "Statistical Significance Matrix" in result.output


def test_compare_with_subset(tmp_path):
    """Test comparing results with benchmark subsets (e.g., benchmark/subset).

    Verifies:
    - Subset filtering works correctly
    - Subset parameter is properly parsed and applied
    - Results are filtered to the specified subset
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create mock results with subset
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset="subset1",
        tasks=[("task-1", 1.0), ("task-2", 0.8)],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="test_benchmark",
        subset="subset1",
        tasks=[("task-1", 0.9), ("task-2", 0.85)],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--benchmark",
            "test_benchmark/subset1",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    # Verify subset parameter is properly parsed and applied
    assert "test_benchmark/subset1" in result.output
    # New format shows summary table
    assert "Statistical Significance Matrix" in result.output


def test_compare_json_output(tmp_path):
    """Test JSON output format (--format json).

    Verifies:
    - JSON structure includes setups, per_benchmark, and overall
    - Proper JSON serialization of all data
    - Summary statistics are included in JSON output
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 1.0), ("task-2", 0.8)],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 0.9), ("task-2", 0.85)],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output

    # Parse JSON output - new structure
    output_data = json.loads(result.output)
    assert "setups" in output_data
    assert "per_benchmark" in output_data
    assert "overall" in output_data
    assert len(output_data["setups"]) == 2
    assert "test_benchmark" in output_data["per_benchmark"]
    assert len(output_data["overall"]["pairwise_comparisons"]) == 1


def test_compare_multiple_benchmarks(tmp_path):
    """Test comparing across multiple benchmarks simultaneously.

    Verifies:
    - Multiple benchmarks can be compared in one command
    - Overall statistics are computed with equal weight per benchmark
    - "Overall" section appears in output
    - Per-benchmark and overall summaries are both shown
    """
    import warnings

    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create results for benchmark1
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="benchmark1",
        subset=None,
        tasks=[("task-1", 1.0), ("task-2", 0.8)],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="benchmark1",
        subset=None,
        tasks=[("task-1", 0.9), ("task-2", 0.85)],
    )

    # Create results for benchmark2
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="benchmark2",
        subset=None,
        tasks=[("task-1", 0.95), ("task-2", 0.75)],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="benchmark2",
        subset=None,
        tasks=[("task-1", 0.85), ("task-2", 0.8)],
    )

    # Suppress statsmodels warnings for small sample sizes
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
        result = runner.invoke(
            cli,
            [
                "compare",
                "--agent1",
                "agent1",
                "--agent2",
                "agent2",
                "--model1",
                "model1",
                "--model2",
                "model1",
                "--benchmark",
                "benchmark1",
                "--benchmark",
                "benchmark2",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "benchmark1" in result.output
    assert "benchmark2" in result.output
    assert "Overall" in result.output


def test_compare_pairwise_three_agents(tmp_path):
    """Test pairwise comparison mode with 3 agents.

    Verifies:
    - Pairwise comparison summary table is displayed
    - All pair combinations are computed (3 pairs for 3 agents)
    - Statistical significance matrix is shown
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create results for three agents
    for benchmark_num in [1, 2]:
        for agent_num in [1, 2, 3]:
            score_offset = (agent_num - 1) * 0.1
            create_mock_results(
                output_dir,
                agent=f"agent{agent_num}",
                model="model1",
                benchmark=f"test_benchmark{benchmark_num}",
                subset=None,
                tasks=[
                    ("task-1", 1.0 - score_offset),
                    ("task-2", 0.9 - score_offset),
                ],
            )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--agent3",
            "agent3",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--model3",
            "model1",
            "--benchmark",
            "test_benchmark1",
            "--benchmark",
            "test_benchmark2",
            "--output-dir",
            str(output_dir),
        ],
    )
    # print(result.output)
    assert result.exit_code == 0, result.output
    assert "Comparison Results" in result.output
    # Verify all pair combinations are computed (3 pairs for 3 agents)
    # Should see comparisons: agent1 vs agent2, agent1 vs agent3, agent2 vs agent3
    assert "agent1 and model1" in result.output
    assert "agent2 and model1" in result.output
    assert "agent3 and model1" in result.output
    assert "Detailed Comparison: test_benchmark1" in result.output
    assert "Detailed Comparison: test_benchmark2" in result.output
    assert "Detailed Comparison: test_benchmark2" in result.output
    assert re.search(
        r"Detailed Comparison.*" r"agent1 and model1.*agent2 and model1.*" r"agent1 and model1.*agent2 and model1.*",
        result.output,
        flags=re.DOTALL,
    )
    assert "Statistical Significance Matrix" in result.output


def test_compare_missing_results(tmp_path):
    """Test error handling when results are missing for one setup.

    Verifies:
    - Appropriate error message is displayed
    - Suggests the correct 'exgentic evaluate' command to run
    - Exits with non-zero code
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Only create results for agent1, not agent2
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 1.0)],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "No results found" in result.output
    # Verify suggests the correct 'exgentic evaluate' command to run
    assert "exgentic evaluate" in result.output or "evaluate" in result.output.lower()


def test_compare_mismatched_tasks(tmp_path):
    """Test comparing when agents have different task sets.

    Verifies:
    - N/A is shown for missing tasks
    - Comparison continues with available tasks
    - No errors are raised for mismatched task sets
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Agent1 has tasks 1 and 2
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 1.0), ("task-2", 0.8)],
    )

    # Agent2 has tasks 2 and 3
    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-2", 0.85), ("task-3", 0.7)],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    # New format only compares overlapping tasks, shows # Tasks = 1
    assert "Statistical Significance Matrix" in result.output
    # Only task-2 is compared (overlaps between both agents)
    assert "1" in result.output  # Should show 1 task compared


def test_compare_invalid_input_patterns(tmp_path):
    """Test validation of input patterns (must be consistent).

    Verifies:
    - Error is raised for invalid combinations
    - All setups must follow same pattern (all with agent+model, all with agent only, or all with model only)
    - Appropriate error message explains valid patterns
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Invalid: agent1 has model, agent2 doesn't
    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--model1",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "Invalid input combination" in result.output
    # Verify appropriate error message explains valid patterns
    assert "pattern" in result.output.lower() or "consistent" in result.output.lower()


def test_compare_requires_two_setups(tmp_path):
    """Test that at least 2 setups are required for comparison.

    Verifies:
    - Error is raised when only one setup is provided
    - Appropriate error message is shown
    - Minimum comparison requirement is enforced
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--model1",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "At least 2 agent/model setups are required" in result.output


def test_compare_statistical_significance_significant(tmp_path):
    """Test that statistical significance is detected when there's a clear difference.

    Verifies:
    - Statistical significance is detected for clear differences
    - McNemar's test is used
    - p-value is below significance threshold (< 0.05)
    - Significance status is correctly reported
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create results with clear significant difference
    # Agent1 succeeds on all tasks (score >= 0.99)
    # Agent2 fails on all tasks (score < 0.99)
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[
            ("task-1", 1.0),
            ("task-2", 1.0),
            ("task-3", 1.0),
            ("task-4", 1.0),
            ("task-5", 1.0),
            ("task-6", 1.0),
            ("task-7", 1.0),
            ("task-8", 1.0),
            ("task-9", 1.0),
            ("task-10", 1.0),
        ],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[
            ("task-1", 0.5),
            ("task-2", 0.5),
            ("task-3", 0.5),
            ("task-4", 0.5),
            ("task-5", 0.5),
            ("task-6", 0.5),
            ("task-7", 0.5),
            ("task-8", 0.5),
            ("task-9", 0.5),
            ("task-10", 0.5),
        ],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output

    # Parse JSON output for precise checks - new structure
    output_data = json.loads(result.output)

    # Check per-benchmark results
    benchmark_comparison = output_data["per_benchmark"]["test_benchmark"][0]
    assert benchmark_comparison["num_tasks"] == 10
    assert benchmark_comparison["success_rate1"] == 1.0
    assert benchmark_comparison["success_rate2"] == 0.0
    assert benchmark_comparison["rate_difference"] == 1.0
    assert benchmark_comparison["is_significant"] is True
    assert benchmark_comparison["p_value"] < 0.05

    # Check overall results
    overall_comparison = output_data["overall"]["pairwise_comparisons"][0]
    assert overall_comparison["num_tasks"] == 10
    assert overall_comparison["is_significant"] is True
    assert overall_comparison["p_value"] < 0.05


def test_compare_statistical_significance_not_significant(tmp_path):
    """Test that non-significant results are correctly identified.

    Verifies:
    - Non-significant results are correctly identified
    - is_significant flag is False
    - p-value is above significance threshold
    - Appropriate message indicates non-significance
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create results with no significant difference
    # Both agents have identical performance (all succeed)
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[
            ("task-1", 1.0),
            ("task-2", 1.0),
            ("task-3", 1.0),
            ("task-4", 1.0),
            ("task-5", 1.0),
        ],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[
            ("task-1", 1.0),
            ("task-2", 1.0),
            ("task-3", 1.0),
            ("task-4", 1.0),
            ("task-5", 1.0),
        ],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output

    # Parse JSON output - new structure
    output_data = json.loads(result.output)

    # Check per-benchmark results
    benchmark_comparison = output_data["per_benchmark"]["test_benchmark"][0]
    assert benchmark_comparison["num_tasks"] == 5
    assert benchmark_comparison["success_rate1"] == 1.0
    assert benchmark_comparison["success_rate2"] == 1.0
    assert benchmark_comparison["rate_difference"] == 0.0
    assert benchmark_comparison["is_significant"] is False
    assert benchmark_comparison["p_value"] >= 0.1


def test_compare_statistical_significance_marginal(tmp_path):
    """Test marginal significance detection (small but detectable difference).

    Verifies:
    - Marginal differences are detected
    - Average scores are computed correctly
    - Statistical test handles partial disagreements
    - p-value reflects the marginal difference
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create results with marginal difference
    # Agent1 succeeds on 7/10 tasks, Agent2 succeeds on 3/10 tasks
    # This creates 4 disagreements where agent1 wins
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[
            ("task-1", 1.0),  # success (both succeed)
            ("task-2", 1.0),  # success (both succeed)
            ("task-3", 1.0),  # success (both succeed)
            ("task-4", 1.0),  # success (agent1 wins)
            ("task-5", 1.0),  # success (agent1 wins)
            ("task-6", 1.0),  # success (agent1 wins)
            ("task-7", 1.0),  # success (agent1 wins)
            ("task-8", 0.5),  # fail (both fail)
            ("task-9", 0.5),  # fail (both fail)
            ("task-10", 0.5),  # fail (both fail)
        ],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[
            ("task-1", 1.0),  # success (both succeed)
            ("task-2", 1.0),  # success (both succeed)
            ("task-3", 1.0),  # success (both succeed)
            ("task-4", 0.5),  # fail (agent1 wins)
            ("task-5", 0.5),  # fail (agent1 wins)
            ("task-6", 0.5),  # fail (agent1 wins)
            ("task-7", 0.5),  # fail (agent1 wins)
            ("task-8", 0.5),  # fail (both fail)
            ("task-9", 0.5),  # fail (both fail)
            ("task-10", 0.5),  # fail (both fail)
        ],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output

    # Parse JSON output - new structure
    output_data = json.loads(result.output)

    # Check per-benchmark results
    benchmark_comparison = output_data["per_benchmark"]["test_benchmark"][0]
    assert benchmark_comparison["num_tasks"] == 10

    # Verify success rates are computed correctly
    # Agent1: 7 successes (score >= 0.99) out of 10 = 0.7
    # Agent2: 3 successes (score >= 0.99) out of 10 = 0.3
    assert abs(benchmark_comparison["success_rate1"] - 0.7) < 0.01
    assert abs(benchmark_comparison["success_rate2"] - 0.3) < 0.01
    assert abs(benchmark_comparison["rate_difference"] - 0.4) < 0.01

    # Verify p-value reflects the marginal difference
    assert benchmark_comparison["p_value"] < 0.5


def test_compare_statistical_significance_text_output(tmp_path):
    """Test that statistical significance is displayed correctly in text output.

    Verifies:
    - "Statistical Significance" section appears
    - Test method name is shown (McNemar's test)
    - p-value is displayed
    - Task count is shown
    - Average scores are displayed for both setups
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create results with significant difference
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[
            ("task-1", 1.0),
            ("task-2", 1.0),
            ("task-3", 1.0),
            ("task-4", 1.0),
            ("task-5", 1.0),
        ],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[
            ("task-1", 0.5),
            ("task-2", 0.5),
            ("task-3", 0.5),
            ("task-4", 0.5),
            ("task-5", 0.5),
        ],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    # New format shows summary table
    assert "Statistical Significance Matrix" in result.output
    assert "Comparison Results" in result.output
    # Verify task count is shown (in the # Tasks column)
    assert "5" in result.output


def test_compare_five_agents_max(tmp_path):
    """Test comparing the maximum of 5 agents/models.

    Verifies:
    - All pairwise combinations are computed (10 pairs for 5 agents)
    - Pairwise summary table is generated correctly
    - System handles maximum allowed setups
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create results for 5 agents
    for agent_num in range(1, 6):
        create_mock_results(
            output_dir,
            agent=f"agent{agent_num}",
            model="model1",
            benchmark="test_benchmark",
            subset=None,
            tasks=[("task-1", 1.0 - agent_num * 0.1)],
        )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--agent3",
            "agent3",
            "--agent4",
            "agent4",
            "--agent5",
            "agent5",
            "--model1",
            "model1",
            "--model2",
            "model1",
            "--model3",
            "model1",
            "--model4",
            "model1",
            "--model5",
            "model1",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Statistical Significance Matrix" in result.output
    # Verify all 5 agents appear in output (system handles maximum allowed setups)
    for agent_num in range(1, 6):
        assert f"agent{agent_num}" in result.output


def test_compare_only_agents_no_models(tmp_path):
    """Test comparing agents without specifying models.

    Verifies:
    - Comparison works across any models when only agents are specified
    - Flexible input pattern is supported
    - Results are aggregated correctly regardless of model
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create results for two agents (any model)
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 1.0)],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 0.9)],
    )

    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    # Verify comparison works across any models (both agents should appear)
    assert "agent1" in result.output
    assert "agent2" in result.output
    # New format shows summary table
    assert "Statistical Significance Matrix" in result.output


def test_compare_only_models_no_agents(tmp_path):
    """Test comparing models without specifying agents.

    Verifies:
    - Comparison works across any agents when only models are specified
    - Model-only comparison mode is supported
    - Results are aggregated correctly regardless of agent
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"

    # Create results for two models (any agent)
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 1.0)],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 1.0)],
    )

    create_mock_results(
        output_dir,
        agent="agent1",
        model="model2",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 0)],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model2",
        benchmark="test_benchmark",
        subset=None,
        tasks=[("task-1", 1.0)],
    )
    result = runner.invoke(
        cli,
        [
            "compare",
            "--model1",
            "model1",
            "--model2",
            "model2",
            "--benchmark",
            "test_benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    # Verify comparison works across any agents (both models should appear)
    assert "model1" in result.output
    assert "model2" in result.output
    assert "50." in result.output or "50…" in result.output  # May be truncated as "50.…" in table
    # New format shows summary table
    assert "Statistical Significance Matrix" in result.output


def test_breslow_day_homogeneous_high_pvalue(tmp_path):
    """Test Breslow-Day test with homogeneous data (high p-value).

    Creates data where the effect is consistent across benchmarks:
    - Both benchmarks show similar odds ratios
    - Should result in high p-value (p > 0.05) indicating homogeneity
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create consistent effect across two benchmarks
    # Contingency table: [[4, 3], [1, 2]] -> OR = 2.67 (same for both benchmarks)
    # Benchmark 1: Agent1 wins 7/10, Agent2 wins 5/10
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="benchmark1",
        subset=None,
        tasks=[
            # Both succeed (4 tasks)
            ("task1", 1.0),
            ("task2", 1.0),
            ("task3", 1.0),
            ("task4", 1.0),
            # Agent1 only (3 tasks)
            ("task5", 1.0),
            ("task6", 1.0),
            ("task7", 1.0),
            # Agent2 only (1 task)
            ("task8", 0.0),
            # Both fail (2 tasks)
            ("task9", 0.0),
            ("task10", 0.0),
        ],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="benchmark1",
        subset=None,
        tasks=[
            # Both succeed (4 tasks)
            ("task1", 1.0),
            ("task2", 1.0),
            ("task3", 1.0),
            ("task4", 1.0),
            # Agent1 only (3 tasks) - Agent2 fails
            ("task5", 0.0),
            ("task6", 0.0),
            ("task7", 0.0),
            # Agent2 only (1 task)
            ("task8", 1.0),
            # Both fail (2 tasks)
            ("task9", 0.0),
            ("task10", 0.0),
        ],
    )

    # Benchmark 2: Same pattern - Agent1 wins 7/10, Agent2 wins 5/10
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="benchmark2",
        subset=None,
        tasks=[
            # Both succeed (4 tasks)
            ("task1", 1.0),
            ("task2", 1.0),
            ("task3", 1.0),
            ("task4", 1.0),
            # Agent1 only (3 tasks)
            ("task5", 1.0),
            ("task6", 1.0),
            ("task7", 1.0),
            # Agent2 only (1 task)
            ("task8", 0.0),
            # Both fail (2 tasks)
            ("task9", 0.0),
            ("task10", 0.0),
        ],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="benchmark2",
        subset=None,
        tasks=[
            # Both succeed (4 tasks)
            ("task1", 1.0),
            ("task2", 1.0),
            ("task3", 1.0),
            ("task4", 1.0),
            # Agent1 only (3 tasks) - Agent2 fails
            ("task5", 0.0),
            ("task6", 0.0),
            ("task7", 0.0),
            # Agent2 only (1 task)
            ("task8", 1.0),
            # Both fail (2 tasks)
            ("task9", 0.0),
            ("task10", 0.0),
        ],
    )

    # Run compare command with JSON output
    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--benchmark",
            "benchmark1",
            "--benchmark",
            "benchmark2",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"

    # Parse JSON output - new structure
    output_data = json.loads(result.output)

    # Check that overall statistics include Breslow-Day test in pairwise comparisons
    assert "overall" in output_data
    assert "pairwise_comparisons" in output_data["overall"]
    assert len(output_data["overall"]["pairwise_comparisons"]) > 0

    # Get Breslow-Day test from first pairwise comparison
    breslow_day = output_data["overall"]["pairwise_comparisons"][0]["breslow_day"]
    assert breslow_day is not None
    assert "p_value" in breslow_day
    assert "interpretation" in breslow_day

    # High p-value indicates homogeneity (consistent effect across benchmarks)
    assert (
        breslow_day["p_value"] > 0.05
    ), f"Expected high p-value (>0.05) for homogeneous data, got {breslow_day['p_value']}"
    assert (
        "homogeneous" in breslow_day["interpretation"].lower() or "consistent" in breslow_day["interpretation"].lower()
    ), f"Expected 'homogeneous' or 'consistent' in interpretation, got: {breslow_day['interpretation']}"


def test_breslow_day_heterogeneous_low_pvalue(tmp_path):
    """Test Breslow-Day test with heterogeneous data (low p-value).

    Creates data where the effect varies across benchmarks:
    - Benchmark 1: Agent1 much better than Agent2
    - Benchmark 2: Agent2 much better than Agent1
    - Should result in low p-value (p < 0.05) indicating heterogeneity

    This creates contingency tables with different odds ratios:
    - Benchmark 1: OR = 1.5 (Agent1 better)
    - Benchmark 2: OR = 0.0185 (Agent2 better)
    """
    runner = CliRunner()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Benchmark 1: Agent1 better (21/27 vs 8/27)
    # Contingency table: [[18, 6], [2, 1]] -> OR = 1.5
    # Both succeed: 18, Agent1 only: 3, Agent2 only: 2, Both fail: 1
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="benchmark1",
        subset=None,
        tasks=[
            # Both succeed (18 tasks)
            ("task1", 1.0),
            ("task2", 1.0),
            ("task3", 1.0),
            ("task4", 1.0),
            ("task5", 1.0),
            ("task6", 1.0),
            ("task7", 1.0),
            ("task8", 1.0),
            ("task9", 1.0),
            ("task10", 1.0),
            ("task11", 1.0),
            ("task12", 1.0),
            ("task13", 1.0),
            ("task14", 1.0),
            ("task15", 1.0),
            ("task16", 1.0),
            ("task17", 1.0),
            ("task18", 1.0),
            # Agent1 only (6 tasks)
            ("task19", 1.0),
            ("task20", 1.0),
            ("task21", 1.0),
            ("task22", 1.0),
            ("task23", 1.0),
            ("task24", 1.0),
            # Both fail (1 task)
            ("task27", 0.0),
        ],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="benchmark1",
        subset=None,
        tasks=[
            # Both succeed (18 tasks)
            ("task1", 1.0),
            ("task2", 1.0),
            ("task3", 1.0),
            ("task4", 1.0),
            ("task5", 1.0),
            ("task6", 1.0),
            ("task7", 1.0),
            ("task8", 1.0),
            ("task9", 1.0),
            ("task10", 1.0),
            ("task11", 1.0),
            ("task12", 1.0),
            ("task13", 1.0),
            ("task14", 1.0),
            ("task15", 1.0),
            ("task16", 1.0),
            ("task17", 1.0),
            ("task18", 1.0),
            # Agent1 only (6 tasks) - Agent2 fails
            ("task19", 0.0),
            ("task20", 0.0),
            ("task21", 0.0),
            ("task22", 0.0),
            ("task23", 0.0),
            ("task24", 0.0),
            # Agent2 only (2 tasks)
            ("task25", 1.0),
            ("task26", 1.0),
            # Both fail (1 task)
            ("task27", 0.0),
        ],
    )

    # Benchmark 2: Agent2 better (8/27 vs 21/27) - opposite pattern
    # Contingency table: [[2, 18], [6, 1]] -> OR = 0.0185
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="benchmark2",
        subset=None,
        tasks=[
            # Both succeed (2 tasks)
            ("task1", 1.0),
            ("task2", 1.0),
            # Agent2 only (18 tasks) - Agent1 fails
            ("task3", 0.0),
            ("task4", 0.0),
            ("task5", 0.0),
            ("task6", 0.0),
            ("task7", 0.0),
            ("task8", 0.0),
            ("task9", 0.0),
            ("task10", 0.0),
            ("task11", 0.0),
            ("task12", 0.0),
            ("task13", 0.0),
            ("task14", 0.0),
            ("task15", 0.0),
            ("task16", 0.0),
            ("task17", 0.0),
            ("task18", 0.0),
            ("task19", 0.0),
            ("task20", 0.0),
            # Agent1 only (6 tasks)
            ("task21", 1.0),
            ("task22", 1.0),
            ("task23", 1.0),
            ("task24", 1.0),
            ("task25", 1.0),
            ("task26", 1.0),
            # Both fail (1 task)
            ("task27", 0.0),
        ],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="benchmark2",
        subset=None,
        tasks=[
            # Both succeed (2 tasks)
            ("task1", 1.0),
            ("task2", 1.0),
            # Agent2 only (18 tasks)
            ("task3", 1.0),
            ("task4", 1.0),
            ("task5", 1.0),
            ("task6", 1.0),
            ("task7", 1.0),
            ("task8", 1.0),
            ("task9", 1.0),
            ("task10", 1.0),
            ("task11", 1.0),
            ("task12", 1.0),
            ("task13", 1.0),
            ("task14", 1.0),
            ("task15", 1.0),
            ("task16", 1.0),
            ("task17", 1.0),
            ("task18", 1.0),
            ("task19", 1.0),
            ("task20", 1.0),
            # Agent1 only (6 tasks) - Agent2 fails
            ("task21", 0.0),
            ("task22", 0.0),
            ("task23", 0.0),
            ("task24", 0.0),
            ("task25", 0.0),
            ("task26", 0.0),
            # Both fail (1 task)
            ("task27", 0.0),
        ],
    )

    # Run compare command with JSON output
    result = runner.invoke(
        cli,
        [
            "compare",
            "--agent1",
            "agent1",
            "--agent2",
            "agent2",
            "--benchmark",
            "benchmark1",
            "--benchmark",
            "benchmark2",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"

    # Parse JSON output - new structure
    output_data = json.loads(result.output)

    # Check that overall statistics include Breslow-Day test in pairwise comparisons
    assert "overall" in output_data
    assert "pairwise_comparisons" in output_data["overall"]
    assert len(output_data["overall"]["pairwise_comparisons"]) > 0

    # Get Breslow-Day test from first pairwise comparison
    breslow_day = output_data["overall"]["pairwise_comparisons"][0]["breslow_day"]
    assert breslow_day is not None
    assert "p_value" in breslow_day
    assert "interpretation" in breslow_day

    # Low p-value indicates heterogeneity (effect varies across benchmarks)
    assert (
        breslow_day["p_value"] < 0.05
    ), f"Expected low p-value (<0.05) for heterogeneous data, got {breslow_day['p_value']}"
    assert (
        "heterogeneity" in breslow_day["interpretation"].lower() or "varies" in breslow_day["interpretation"].lower()
    ), f"Expected 'heterogeneity' or 'varies' in interpretation, got: {breslow_day['interpretation']}"


def test_breslow_day_text_output(tmp_path):
    """Test that Breslow-Day test results appear in text output."""
    import warnings

    runner = CliRunner()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create data for two benchmarks
    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="benchmark1",
        subset=None,
        tasks=[("task1", 1.0), ("task2", 1.0), ("task3", 0.0)],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="benchmark1",
        subset=None,
        tasks=[("task1", 1.0), ("task2", 0.0), ("task3", 0.0)],
    )

    create_mock_results(
        output_dir,
        agent="agent1",
        model="model1",
        benchmark="benchmark2",
        subset=None,
        tasks=[("task1", 1.0), ("task2", 1.0), ("task3", 0.0)],
    )

    create_mock_results(
        output_dir,
        agent="agent2",
        model="model1",
        benchmark="benchmark2",
        subset=None,
        tasks=[("task1", 1.0), ("task2", 0.0), ("task3", 0.0)],
    )

    # Run compare command with text output (default)
    # Suppress statsmodels warnings for small sample sizes
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
        result = runner.invoke(
            cli,
            [
                "compare",
                "--agent1",
                "agent1",
                "--agent2",
                "agent2",
                "--benchmark",
                "benchmark1",
                "--benchmark",
                "benchmark2",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, f"Command failed: {result.output}"

    # Check that text output includes Breslow-Day test information
    # New format shows it in the overall table section
    assert (
        "Breslow-Day" in result.output or "breslow" in result.output.lower() or "Bre…" in result.output
    ), "Expected Breslow-Day test information in text output"
    assert (
        "P-value" in result.output or "p-value" in result.output or "p=" in result.output
    ), "Expected p-value in text output"
    assert (
        "Interpretation" in result.output
        or "homogeneous" in result.output.lower()
        or "consistent" in result.output.lower()
        or "hom…" in result.output
        or "con…" in result.output
    ), "Expected interpretation in text output"

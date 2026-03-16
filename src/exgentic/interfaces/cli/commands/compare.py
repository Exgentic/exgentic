# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple, Union

import rich_click as click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from ....core.types import SessionResults

stats = None
StratifiedTable = None


def _ensure_compare_deps() -> None:
    global stats, StratifiedTable
    if stats is None or StratifiedTable is None:
        try:
            from scipy import stats as _stats
            from statsmodels.stats.contingency_tables import StratifiedTable as _StratifiedTable
        except ImportError as exc:
            raise click.ClickException(
                "Compare commands require the optional analysis dependencies. "
                "Install them with `pip install 'exgentic[analysis]'`."
            ) from exc

        stats = _stats
        StratifiedTable = _StratifiedTable


class TaskInfo(NamedTuple):
    """Task information including session result, setup info, and file path."""

    session_result: SessionResults
    setup_info: str  # Agent name or model name when comparing across agents/models
    file_path: str  # Path to the results.json file


class BenchmarkStats(NamedTuple):
    """Statistics for a single benchmark comparison between two setups."""

    benchmark_name: str
    setup1_label: str
    setup2_label: str
    success_rate1: float
    success_rate2: float
    rate_diff: float
    p_value: float
    num_tasks: int
    is_significant: bool
    winner: str
    success1: list[bool]  # For Breslow-Day test
    success2: list[bool]  # For Breslow-Day test
    breslow_day_pvalue: Optional[float] = 0  # Breslow-Day test p-value (if multiple benchmarks)
    breslow_day_interpretation: Optional[str] = ""  # Breslow-Day test interpretation


class PairwiseComparison(NamedTuple):
    """Complete pairwise comparison data between two setups."""

    setup1_label: str
    setup2_label: str
    per_benchmark: dict[str, BenchmarkStats]  # benchmark_name -> stats
    aggregate_stats: Optional[BenchmarkStats]  # Overall stats across benchmarks


# Type alias for composite key: either a string (task_id) or tuple (task_id, setup_info)
CompositeKey = Union[str, Tuple[str, str]]


def _parse_benchmark_spec(benchmark_spec: str) -> tuple[str, str | None, int | None]:
    """Parse benchmark specification in format 'benchmark', 'benchmark/subset', or 'benchmark/subset=limit'.

    Examples:
        'tau' -> ('tau', None, None)
        'tau/retail' -> ('tau', 'retail', None)
        'tau/retail=30' -> ('tau', 'retail', 30)
        'gsm8k=100' -> ('gsm8k', None, 100)

    Returns:
        Tuple of (benchmark_name, subset_name or None, limit or None)
    """
    # Check for limit parameter (=N)
    limit = None
    if "=" in benchmark_spec:
        benchmark_spec, limit_str = benchmark_spec.rsplit("=", 1)
        try:
            limit = int(limit_str)
        except ValueError:
            raise click.ClickException(
                f"Invalid limit value in benchmark spec: {limit_str}. Must be an integer."
            ) from None

    # Check for subset (/subset)
    if "/" in benchmark_spec:
        parts = benchmark_spec.split("/", 1)
        return parts[0], parts[1], limit
    return benchmark_spec, None, limit


def _load_run_results(
    output_dir: str,
    agent: str | None,
    model: str | None,
    benchmark: str,
    subset: str | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    benchmark_kwargs: dict[str, Any] | None = None,
    progress: Progress | None = None,
    task_id: Any = None,
) -> dict[CompositeKey, TaskInfo] | None:
    """Load run results for a specific agent/model/benchmark combination by searching config.json files.

    If agent is None, loads results from all agents and stores agent name in setup_info.
    If model is None, loads results from all models and stores model name in setup_info.

    Args:
        output_dir: Directory containing run outputs
        agent: Agent name to filter by, or None for all agents
        model: Model name to filter by, or None for all models
        benchmark: Benchmark name to search for
        subset: Optional benchmark subset
        agent_kwargs: Optional agent configuration parameters
        benchmark_kwargs: Optional benchmark configuration parameters
        progress: Optional Rich Progress instance for showing progress
        task_id: Optional task ID for updating progress

    Returns:
        Dict mapping composite_key (task_id or (task_id, setup_info)) -> TaskInfo, or None if not found
    """
    output_path = Path(output_dir)
    results_by_task: dict[CompositeKey, TaskInfo] = {}

    # Collect all config files first to show progress
    config_files = list(output_path.glob("**/config.json"))

    # Search through all config.json files in the output directory
    for idx, config_file in enumerate(config_files):
        if progress and task_id is not None:
            progress.update(task_id, completed=idx, total=len(config_files))
        try:
            with open(config_file, encoding="utf-8") as f:
                config_data = json.load(f)

            # Handle two different config.json formats:
            # 1. Direct format: {"benchmark": "...", "agent": "...", ...}
            # 2. Nested format: {"benchmark": {"slug_name": "..."}, "agent": {"slug_name": "..."}, ...}

            # Extract benchmark name
            config_benchmark = config_data.get("benchmark")
            if isinstance(config_benchmark, dict):
                config_benchmark = config_benchmark.get("slug_name")

            # Extract agent name
            config_agent = config_data.get("agent")
            if isinstance(config_agent, dict):
                config_agent = config_agent.get("slug_name")

            # Extract model name
            config_model = config_data.get("model")
            if config_model is None and isinstance(config_data.get("agent"), dict):
                config_model = config_data.get("agent", {}).get("model_name")

            # Extract subset name
            config_subset = config_data.get("subset")
            if config_subset is None and isinstance(config_data.get("benchmark"), dict):
                config_subset = config_data.get("benchmark", {}).get("params", {}).get("subset")

            # Check if this config file matches our criteria
            # Check benchmark
            if config_benchmark != benchmark:
                continue

            # Check agent if specified
            if agent is not None:
                if config_agent != agent:
                    continue

            # Check model if specified
            if model is not None:
                if config_model is None:
                    continue
                # Normalize model names for comparison (handle different formats)
                if model not in config_model and config_model not in model:
                    continue

            # Check subset if specified
            if subset is not None:
                if config_subset != subset:
                    continue

            # If we get here, this is a match
            if progress and task_id is not None:
                progress.update(task_id, description=f"Loading from {config_file.parent.name}...")

            # Determine the run directory based on config location
            # Config can be at: outputs/config.json or outputs/{run_id}/run/config.json
            if config_file.name == "config.json" and config_file.parent.name == "run":
                run_dir = config_file.parent.parent
            elif config_file.parent == output_path:
                # Top-level config.json, need to find the actual run directory
                run_id = config_data.get("run_id")
                if run_id:
                    run_dir = output_path / run_id
                else:
                    continue
            else:
                # Assume config is in the run directory
                run_dir = config_file.parent

            sessions_dir = run_dir / "sessions"

            if not sessions_dir.exists():
                continue

            # Iterate through all session directories
            for session_dir in sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                session_results_file = session_dir / "results.json"
                if not session_results_file.exists():
                    continue

                try:
                    with open(session_results_file, encoding="utf-8") as f:
                        session_data = json.load(f)

                    # Validate session result and ensure success is consistent with score
                    session_result = SessionResults.model_validate(session_data)

                    # Check if success field is consistent with score (success should be True iff score == 1)
                    expected_success = session_result.score == 1.0
                    if session_result.success != expected_success:
                        # console = Console()
                        # console.print(
                        #    f"[yellow]Warning: Inconsistent success field in {session_results_file}[/yellow]\n"
                        #    f"  Session ID: {session_result.session_id}\n"
                        #    f"  Score: {session_result.score}\n"
                        #    f"  Success field: {session_result.success}\n"
                        #    f"  Expected success: {expected_success}\n"
                        #    f"  Correcting success to match score..."
                        # )
                        # Correct the success field to match the score
                        session_result.success = expected_success
                    session_task_id = session_result.task_id
                    if session_task_id == "" or session_task_id is None:
                        raise ValueError(
                            f"Illegal task key '{session_task_id}'!\n" f"    File: {session_results_file}\n"
                        )

                    # Determine setup info based on what's being compared
                    setup_info = ""
                    if agent is None and config_agent:
                        setup_info = config_agent
                    elif model is None and config_model:
                        setup_info = config_model

                    # Create composite key: (task_id, setup_info) for unique identification
                    # When setup_info is empty (specific agent/model), just use task_id
                    composite_key = (session_task_id, setup_info) if setup_info else session_task_id

                    # Check for duplicate composite keys
                    if composite_key in results_by_task:
                        existing_task_info = results_by_task[composite_key]
                        if existing_task_info.session_result.success != session_result.success:
                            raise ValueError(
                                f"Duplicate entry found for task '{session_task_id}' with setup '{setup_info}'!\n"
                                f"  First occurrence:\n"
                                f"    File: {existing_task_info.file_path}\n"
                                f"    Session ID: {existing_task_info.session_result.session_id}\n"
                                f"    Success: {existing_task_info.session_result.success}\n"
                                f"  Second occurrence:\n"
                                f"    File: {session_results_file}\n"
                                f"    Session ID: {session_result.session_id}\n"
                                f"    Success: {session_result.success}\n"
                                f"  This indicates overlapping results."
                            )

                    task_info = TaskInfo(
                        session_result=session_result,
                        setup_info=setup_info,
                        file_path=str(session_results_file),
                    )
                    results_by_task[composite_key] = task_info
                    # print(f"Loaded session {session_result.session_id} with composite_key {composite_key} "
                    #       f"with score {session_result.score} from file {session_results_file}.")

                except Exception as e:
                    # Print error but continue loading other sessions
                    console = Console()
                    console.print(f"[yellow]Warning: Error loading {session_results_file}: {e}[/yellow]")
                    continue

            # When agent or model is None, continue searching for more runs
            # When both are specified, return after first match
            # if config_agent is not None and config_model is not None and results_by_task:
            #    if progress and task_id is not None:
            #        progress.update(task_id, completed=len(config_files),
            #                        description=f"Loaded {len(results_by_task)} sessions")
            #    return results_by_task

        except Exception as e:
            # Print error but continue with other config files
            console = Console()
            console.print(f"[yellow]Warning: Error reading {config_file}: {e}[/yellow]")
            continue

    # Complete progress
    if progress and task_id is not None:
        progress.update(
            task_id,
            completed=len(config_files),
            description=f"Loaded {len(results_by_task)} sessions",
        )

    # Return accumulated results (for agent=None or model=None case) or None if nothing found
    return results_by_task if results_by_task else None


def _calculate_benchmark_stats(
    benchmark_name: str,
    setup1_label: str,
    setup2_label: str,
    setup1_results: dict[CompositeKey, TaskInfo],
    setup2_results: dict[CompositeKey, TaskInfo],
    limit: int | None = None,
) -> Optional[BenchmarkStats]:
    """Calculate statistics for comparing two setups on a single benchmark.

    Args:
        benchmark_name: Name of the benchmark
        setup1_label: Label for first setup
        setup2_label: Label for second setup
        setup1_results: Results for first setup
        setup2_results: Results for second setup
        limit: Optional limit on number of tasks to compare

    Returns:
        BenchmarkStats object or None if no valid comparisons
    """
    # Get comparison data
    benchmark_data = _compare_results(benchmark_name, setup1_results, setup2_results)

    # Extract valid comparisons (tasks present in both setups)
    valid_comparisons = [(s1, s2) for _, _, _, _, s1, s2 in benchmark_data if s1 is not None and s2 is not None]

    # Apply limit if specified
    if limit is not None and len(valid_comparisons) > limit:
        valid_comparisons = valid_comparisons[:limit]

    if not valid_comparisons:
        return None

    # Extract success lists
    success1 = [s1 for s1, _ in valid_comparisons]
    success2 = [s2 for _, s2 in valid_comparisons]

    # Calculate rates
    success_rate1 = sum(success1) / len(success1)
    success_rate2 = sum(success2) / len(success2)
    num_tasks = len(valid_comparisons)
    rate_diff = success_rate1 - success_rate2

    # Compute statistical significance
    (
        statistic,
        p_value,
        significance,
        method,
        n01,
        n10,
    ) = _compute_statistical_significance_mcnemar(success1, success2)

    is_significant = p_value < 0.05
    winner = setup1_label if rate_diff > 0 else setup2_label if rate_diff < 0 else "Tie"

    return BenchmarkStats(
        benchmark_name=benchmark_name,
        setup1_label=setup1_label,
        setup2_label=setup2_label,
        success_rate1=success_rate1,
        success_rate2=success_rate2,
        rate_diff=rate_diff,
        p_value=p_value,
        num_tasks=num_tasks,
        is_significant=is_significant,
        winner=winner,
        success1=success1,
        success2=success2,
    )


def _calculate_aggregate_stats(
    setup1_label: str,
    setup2_label: str,
    per_benchmark_stats: list[BenchmarkStats],
) -> Optional[BenchmarkStats]:
    """Calculate aggregate statistics across multiple benchmarks.

    Args:
        setup1_label: Label for first setup
        setup2_label: Label for second setup
        per_benchmark_stats: List of BenchmarkStats from individual benchmarks

    Returns:
        BenchmarkStats object with aggregated data or None if no data
    """
    if not per_benchmark_stats:
        return None

    # Aggregate success lists from all benchmarks
    all_success1 = []
    all_success2 = []
    for bench_stats in per_benchmark_stats:
        all_success1.extend(bench_stats.success1)
        all_success2.extend(bench_stats.success2)

    if not all_success1:
        return None

    # Calculate aggregate rates
    success_rate1 = sum(all_success1) / len(all_success1)
    success_rate2 = sum(all_success2) / len(all_success2)
    num_tasks = len(all_success1)
    rate_diff = success_rate1 - success_rate2

    # Compute statistical significance
    (
        statistic,
        p_value,
        significance,
        method,
        n01,
        n10,
    ) = _compute_statistical_significance_mcnemar(all_success1, all_success2)

    is_significant = p_value < 0.05
    winner = setup1_label if rate_diff > 0 else setup2_label if rate_diff < 0 else "Tie"

    if len(per_benchmark_stats) > 1:
        bd_data = [(bench_stats.success1, bench_stats.success2) for bench_stats in per_benchmark_stats]
        bd_p_value, bd_interp = _compute_breslow_day_test(bd_data)
    else:
        bd_p_value, bd_interp = None, None
    return BenchmarkStats(
        benchmark_name="Overall",
        setup1_label=setup1_label,
        setup2_label=setup2_label,
        success_rate1=success_rate1,
        success_rate2=success_rate2,
        rate_diff=rate_diff,
        p_value=p_value,
        num_tasks=num_tasks,
        is_significant=is_significant,
        winner=winner,
        success1=all_success1,
        success2=all_success2,
        breslow_day_pvalue=bd_p_value,
        breslow_day_interpretation=bd_interp,
    )


def _compare_results(
    benchmark: str,
    setup1_results: dict[CompositeKey, TaskInfo] | None,
    setup2_results: dict[CompositeKey, TaskInfo] | None,
) -> list[tuple[str, str, str, str, bool | None, bool | None]]:
    """Compare results from two setups for a single benchmark.

    Compares results by matching composite keys directly. For cross-agent/model comparisons,
    this means comparing (task_id, setup_info) pairs.

    Returns:
        List of (benchmark, task_id, setup1_info, setup2_info, setup1_success, setup2_success)
    """
    comparison_data = []

    if setup1_results is None and setup2_results is None:
        return comparison_data

    setup1_tasks = setup1_results or {}
    setup2_tasks = setup2_results or {}

    # Get all composite keys from both setups
    all_keys = set(setup1_tasks.keys()) | set(setup2_tasks.keys())

    # Remove None or empty keys
    all_keys = {k for k in all_keys if k is not None and k != ""}

    for composite_key in sorted(all_keys):
        # Extract task_id from composite key
        task_id = composite_key[0] if isinstance(composite_key, tuple) else composite_key

        setup1_task_info = setup1_tasks.get(composite_key)
        setup2_task_info = setup2_tasks.get(composite_key)

        setup1_success = setup1_task_info.session_result.success if setup1_task_info else None
        setup2_success = setup2_task_info.session_result.success if setup2_task_info else None
        setup1_info = setup1_task_info.setup_info if setup1_task_info else ""
        setup2_info = setup2_task_info.setup_info if setup2_task_info else ""

        comparison_data.append(
            (
                benchmark,
                task_id,
                setup1_info,
                setup2_info,
                setup1_success,
                setup2_success,
            )
        )

    return comparison_data


def _compute_statistical_significance_mcnemar(
    success1: list[bool],
    success2: list[bool],
) -> tuple[float, float, str, str, int, int]:
    """Compute statistical significance using McNemar's test for binary outcomes.

    McNemar's test is appropriate for comparing two classifiers on the same test set.
    It tests whether the disagreements between the two classifiers are systematic.

    Args:
        success1: Success outcomes from setup 1 (list of booleans)
        success2: Success outcomes from setup 2 (list of booleans)

    Returns:
        Tuple of (statistic, p_value, interpretation, method_name, n01, n10)
    """
    if len(success1) < 2 or len(success2) < 2:
        return 0.0, 1.0, "insufficient data", "McNemar's test", 0, 0

    if len(success1) != len(success2):
        return 0.0, 1.0, "mismatched sample sizes", "McNemar's test", 0, 0

    # Build contingency table
    # n01: setup1 failed, setup2 succeeded
    # n10: setup1 succeeded, setup2 failed
    n01 = sum(1 for s1, s2 in zip(success1, success2) if not s1 and s2)
    n10 = sum(1 for s1, s2 in zip(success1, success2) if s1 and not s2)

    # McNemar's test statistic
    # Use continuity correction for small samples
    if n01 + n10 == 0:
        return 0.0, 1.0, "no disagreements", "McNemar's test", n01, n10

    # Chi-square statistic with continuity correction
    statistic = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)

    # p-value from chi-square distribution with 1 degree of freedom
    # Convert to native Python float for JSON serialization
    p_value = float(1 - stats.chi2.cdf(statistic, df=1))

    # Interpret the result
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    elif p_value < 0.1:
        significance = "marginally significant (p < 0.1)"
    else:
        significance = "not significant (p >= 0.1)"

    return float(statistic), p_value, significance, "McNemar's test", n01, n10


def _compute_breslow_day_test(
    contingency_tables_data: list[tuple[list[bool], list[bool]]],
) -> tuple[float, str]:
    """Compute Breslow-Day test for homogeneity of odds ratios across strata (benchmarks).

    The Breslow-Day test checks whether the odds ratios are consistent across different
    benchmarks. A significant result suggests that the effect of one setup vs another
    varies across benchmarks.

    Args:
        contingency_tables_data: List of (success1, success2) tuples for each benchmark,
                                where success1/success2 are lists of boolean success values

    Returns:
        Tuple of (p_value, interpretation)
    """
    if len(contingency_tables_data) < 2:
        return 1.0, "insufficient benchmarks (need at least 2)"

    # Build 2x2xK contingency table where K is the number of benchmarks
    tables = []
    for success1, success2 in contingency_tables_data:
        if len(success1) != len(success2):
            continue

        # Build 2x2 table for this benchmark
        # Rows: Setup 1 (success=1, failure=0)
        # Cols: Setup 2 (success=1, failure=0)
        n11 = sum(1 for s1, s2 in zip(success1, success2) if s1 and s2)
        n10 = sum(1 for s1, s2 in zip(success1, success2) if s1 and not s2)
        n01 = sum(1 for s1, s2 in zip(success1, success2) if not s1 and s2)
        n00 = sum(1 for s1, s2 in zip(success1, success2) if not s1 and not s2)

        # Create 2x2 table: [[n11, n10], [n01, n00]]
        table = [[n11, n10], [n01, n00]]
        tables.append(table)

    if len(tables) < 2:
        return 1.0, "insufficient valid benchmarks"

    try:
        # Create StratifiedTable and run test_equal_odds (Breslow-Day test)
        st = StratifiedTable(tables)
        result = st.test_equal_odds()
        p_value = float(result.pvalue)

        # Interpret the result
        if p_value < 0.01:
            interpretation = (
                "significant heterogeneity (p < 0.01) - relative success rates varies significantly across benchmarks"
            )
        elif p_value < 0.05:
            interpretation = "moderate heterogeneity (p < 0.05) - relative success rate vary across benchmarks"
        elif p_value < 0.1:
            interpretation = "marginal heterogeneity (p < 0.1) - relative success rate somewhat vary across benchmarks"
        else:
            interpretation = "homogeneous (p >= 0.1) - relative success rate is consistent across benchmarks"

        return p_value, interpretation
    except Exception as e:
        return 1.0, f"test failed: {e!s}"


def _render_significance_matrix(
    benchmark_stats_list: list[BenchmarkStats],
    title: str = "Statistical Significance Matrix",
    benchmark_name: str = "",
) -> None:
    """Render a matrix showing p-value significance levels between setups.

    Matrix format:
    - Rows: Setup 1
    - Columns: Setup 2
    - Cell values: Significance level
      * = p < 0.05
      ** = p < 0.01
      *** = p < 0.001
      - = not significant

    Args:
        benchmark_stats_list: List of BenchmarkStats objects
        title: Title for the matrix
        benchmark_name: Optional benchmark name to include in table title
    """
    console = Console()

    # Extract all unique setups and their success rates
    setup_rates: dict[str, float] = {}
    for bench_stats in benchmark_stats_list:
        if bench_stats.setup1_label not in setup_rates:
            setup_rates[bench_stats.setup1_label] = bench_stats.success_rate1
        if bench_stats.setup2_label not in setup_rates:
            setup_rates[bench_stats.setup2_label] = bench_stats.success_rate2

    # Sort setups by success rate (descending - best first)
    sorted_setups = sorted(setup_rates.keys(), key=lambda s: setup_rates[s], reverse=True)

    # Build p-value matrix and winner matrix
    # Key: (setup1, setup2), Value: (p_value, winner)
    comparison_data: dict[tuple[str, str], tuple[float, str]] = {}

    for bench_stats in benchmark_stats_list:
        comparison_data[(bench_stats.setup1_label, bench_stats.setup2_label)] = (
            bench_stats.p_value,
            bench_stats.winner if bench_stats.is_significant else "",
        )

    # Create the matrix table
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, - not significant\n")

    # Include benchmark name in table title if provided
    table_title = f"P-value Significance Matrix: {benchmark_name}" if benchmark_name else "P-value Significance Matrix"
    table = Table(show_header=True, show_lines=True, title=table_title)

    # Add header row
    table.add_column("Setup", style="cyan", no_wrap=True)
    for setup in sorted_setups:
        table.add_column(setup, style="white", justify="center", no_wrap=True)

    # Add data rows
    for row_setup in sorted_setups:
        row_data = [row_setup]
        for col_setup in sorted_setups:
            if row_setup == col_setup:
                # Diagonal: same setup
                cell = "[dim]-[/dim]"
            else:
                # Look up comparison data for this pair
                data = comparison_data.get((row_setup, col_setup))

                if data is None:
                    # Try reverse lookup (col_setup vs row_setup)
                    data = comparison_data.get((col_setup, row_setup))
                    if data is not None:
                        # Reverse the winner perspective
                        p_value, winner = data
                        # Winner stays the same, but we interpret from row's perspective
                    else:
                        p_value, winner = None, ""
                else:
                    p_value, winner = data

                if p_value is None:
                    cell = "[dim]N/A[/dim]"
                elif p_value < 0.001:
                    if winner == row_setup:
                        cell = "[green]***[/green]"
                    elif winner == col_setup:
                        cell = "[bright_red]***[/bright_red]"
                    else:
                        cell = "***"
                elif p_value < 0.01:
                    if winner == row_setup:
                        cell = "[green]**[/green]"
                    elif winner == col_setup:
                        cell = "[bright_red]**[/bright_red]"
                    else:
                        cell = "**"
                elif p_value < 0.05:
                    if winner == row_setup:
                        cell = "[green]*[/green]"
                    elif winner == col_setup:
                        cell = "[bright_red]*[/bright_red]"
                    else:
                        cell = "*"
                else:
                    cell = "[dim]-[/dim]"

            row_data.append(cell)

        table.add_row(*row_data)

    console.print(table)
    console.print("\n[dim]Green: row setup is significantly better | Red: column setup is significantly better[/dim]")


def _render_pairwise_summary_table(
    benchmark_stats_list: list[BenchmarkStats],
    title: str = "Pairwise Comparison Summary",
) -> None:
    """Render pairwise comparison summary table showing which setups are significantly better.

    Args:
        benchmark_stats_list: List of BenchmarkStats objects
        title: Title for the table
    """
    console = Console()

    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("Shows which setups are statistically significantly better than others (p < 0.05)\n")

    # First, render the significance matrix
    # Extract benchmark name from title if present (format: "Comparison Results: benchmark_name")
    benchmark_name = title.split(": ", 1)[1] if ": " in title else ""
    _render_significance_matrix(
        benchmark_stats_list,
        title="Statistical Significance Matrix",
        benchmark_name=benchmark_name,
    )

    # Sort by Success Rate 1 (descending), then by Success Rate 2 (descending)
    sorted_results = sorted(benchmark_stats_list, key=lambda x: (-x.success_rate1, -x.success_rate2))

    # Include benchmark name in detailed table title if present
    detailed_table_title = f"Detailed Comparison: {benchmark_name}" if benchmark_name else "Detailed Comparison"
    table = Table(title=detailed_table_title, show_lines=True)
    table.add_column("Setup 1", style="cyan", no_wrap=True)
    table.add_column("Setup 2", style="magenta", no_wrap=True)
    table.add_column("Success Rate 1", style="green", justify="right")
    table.add_column("Success Rate 2", style="blue", justify="right")
    table.add_column("Rate Diff", style="yellow", justify="right")
    table.add_column("# Tasks", style="white", justify="right")
    table.add_column("p-value", style="white", justify="right")
    table.add_column("Significant?", style="white", justify="center")
    table.add_column("Winner", style="green", no_wrap=True)
    table.add_column("Breslow-Day", style="white", justify="center")

    for bench_stats in sorted_results:
        rate1_str = f"{bench_stats.success_rate1:.1%}"
        rate2_str = f"{bench_stats.success_rate2:.1%}"

        diff_str = f"{bench_stats.rate_diff:+.1%}"
        if bench_stats.rate_diff > 0:
            diff_str = f"[green]{diff_str}[/green]"
        elif bench_stats.rate_diff < 0:
            diff_str = f"[bright_red]{diff_str}[/bright_red]"

        p_value_str = f"{bench_stats.p_value:.4f}"

        if bench_stats.is_significant:
            sig_str = "[green]✓ Yes[/green]"
            winner_str = f"[green]{bench_stats.winner}[/green]"
        else:
            sig_str = "[yellow]○ No[/yellow]"
            winner_str = "[dim]No difference[/dim]"

        row_data = [
            bench_stats.setup1_label,
            bench_stats.setup2_label,
            rate1_str,
            rate2_str,
            diff_str,
            str(bench_stats.num_tasks),
            p_value_str,
            sig_str,
            winner_str,
        ]

        if bench_stats.breslow_day_interpretation is not None and bench_stats.breslow_day_pvalue is not None:
            if bench_stats.breslow_day_pvalue < 0.05:
                bd_str = f"[yellow]p={bench_stats.breslow_day_pvalue:.3f}[/yellow]"
            else:
                bd_str = f"[green]p={bench_stats.breslow_day_pvalue:.3f}[/green]"
            bd_str += f"\n{bench_stats.breslow_day_interpretation}"
            row_data.append(bd_str)

        table.add_row(*row_data)

    console.print(table)


@click.command("compare")
@click.option("--agent1", help="First agent slug name (optional if comparing across agents)")
@click.option("--agent2", help="Second agent slug name (optional if comparing across agents)")
@click.option("--agent3", help="Third agent slug name (optional)")
@click.option("--agent4", help="Fourth agent slug name (optional)")
@click.option("--agent5", help="Fifth agent slug name (optional)")
@click.option("--model1", help="Model for first agent (optional)")
@click.option("--model2", help="Model for second agent (optional)")
@click.option("--model3", help="Model for third agent (optional)")
@click.option("--model4", help="Model for fourth agent (optional)")
@click.option("--model5", help="Model for fifth agent (optional)")
@click.option(
    "--benchmark",
    "benchmarks",
    multiple=True,
    required=True,
    help="Benchmark(s) to compare in format 'benchmark', 'benchmark/subset', or 'benchmark/subset=limit' "
    "(e.g., tau2/airline, gsm8k, tau2/retail=30 to limit to 30 instances)",
)
@click.option(
    "--output-dir",
    default="./outputs",
    show_default=True,
    help="Output directory to search for results",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def compare_cmd(
    agent1: str | None,
    agent2: str | None,
    agent3: str | None,
    agent4: str | None,
    agent5: str | None,
    model1: str | None,
    model2: str | None,
    model3: str | None,
    model4: str | None,
    model5: str | None,
    benchmarks: tuple[str, ...],
    output_dir: str,
    output_format: str,
) -> None:
    r"""Compare performance of multiple agent/model combinations across benchmarks.

    This command searches the output directory to locate results for each agent/model/benchmark combination.

    Benchmarks can be specified with optional subsets using the format 'benchmark/subset'.

    When comparing 2 setups: performs direct comparison.
    When comparing 3+ setups: performs pairwise comparisons and shows significance matrix.

    Examples:
        # Compare two agents and model configuration on a specific benchmark
        exgentic compare --agent1 tool_calling  --model1 openai/gcp/gemini-3-pro-preview \\
                         --agent2 claude_code  --model1 openai/aws/claude-opus-4-5

        # Compare multiple agents (pairwise)
        exgentic compare --agent1 tool_calling --agent2 claude_code --agent3   openai_solo \\
                         --benchmark gsm8k

        # Compare multiple models
        exgentic compare --model1 openai/aws/claude-opus-4-5  --model2 openai/gcp/gemini-3-pro-preview  \\
                         --benchmark tau2/airline

        # Compare up to 5 different setups
        exgentic compare --agent1 tool_calling  --model1 openai/aws/claude-opus-4-5t \\
                         --agent2 claude_code --model2 openai/gcp/gemini-3-pro-preview \\
                         --agent3 openai_sologemini --model3 openai/gcp/gemini-3-pro-preview \\
                         --benchmark gsm8k --benchmark tau2/airline
    """
    _ensure_compare_deps()

    if not benchmarks:
        raise click.ClickException("At least one --benchmark is required.")

    # Collect all agent/model pairs
    agents = [agent1, agent2, agent3, agent4, agent5]
    models = [model1, model2, model3, model4, model5]

    # Filter out None values and create setup list
    setups = []
    for i, (agent, model) in enumerate(zip(agents, models), 1):
        if agent is not None or model is not None:
            setups.append((i, agent, model))

    if len(setups) < 2:
        raise click.ClickException("At least 2 agent/model setups are required for comparison.")

    if len(setups) > 5:
        raise click.ClickException("Maximum 5 agent/model setups are supported.")

    # Validate input consistency: must be one of three patterns
    # 1. All agent/model pairs defined (both agent and model for each setup)
    # 2. Only agents defined (all setups have agent, no models)
    # 3. Only models defined (all setups have model, no agents)

    has_agents = [agent is not None for _, agent, _ in setups]
    has_models = [model is not None for _, _, model in setups]

    all_have_agents = all(has_agents)
    all_have_models = all(has_models)
    none_have_agents = not any(has_agents)
    none_have_models = not any(has_models)

    # Valid patterns:
    # 1. All have both agent and model
    # 2. All have agent, none have model
    # 3. None have agent, all have model

    if not (
        (all_have_agents and all_have_models)
        or (all_have_agents and none_have_models)
        or (none_have_agents and all_have_models)
    ):
        raise click.ClickException(
            "Invalid input combination. Must use one of these patterns:\n"
            "  1. All setups with both agent and model (e.g., --agent1 X --model1 Y --agent2 A --model2 B)\n"
            "  2. All setups with only agents (e.g., --agent1 X --agent2 Y --agent3 Z)\n"
            "  3. All setups with only models (e.g., --model1 X --model2 Y --model3 Z)"
        )

    # Create labels for each setup
    setup_labels = []
    for idx, agent, model in setups:
        label = (f"{agent}" if agent else "") + (" and " if agent and model else "") + (f"{model}" if model else "")
        if not label:
            label = f"Setup {idx}"
        setup_labels.append(label)

    # Load results for all setups across all benchmarks with progress bar
    all_setup_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=Console(),
    ) as progress:
        for setup_idx, (_idx, agent, model) in enumerate(setups):
            setup_label = setup_labels[setup_idx]
            setup_results = {}

            for benchmark_spec in benchmarks:
                benchmark_name, subset, limit = _parse_benchmark_spec(benchmark_spec)
                display_name = benchmark_spec

                # Create progress task for this load operation
                task_id = progress.add_task(f"Loading {setup_label} on {display_name}...", total=100)

                results = _load_run_results(
                    output_dir,
                    agent,
                    model,
                    benchmark_name,
                    subset,
                    progress=progress,
                    task_id=task_id,
                )

                # Remove the task after completion
                progress.remove_task(task_id)

                if results is None:
                    eval_cmd = "exgentic evaluate"
                    if agent:
                        eval_cmd += f" --agent {agent}"
                    if model:
                        eval_cmd += f" --model {model}"
                    eval_cmd += f" --benchmark {benchmark_name}"
                    if subset:
                        eval_cmd += f" --subset {subset}"

                    raise click.ClickException(
                        f"No results found for {setup_label} on '{benchmark_spec}'. " f"Run '{eval_cmd}' first."
                    )

                setup_results[display_name] = results

            all_setup_results.append(setup_results)

    # Calculate all pairwise comparisons using new calculation functions
    all_pairwise_comparisons: list[PairwiseComparison] = []

    # Compare all pairs (i, j) where i < j
    for i in range(len(setups)):
        for j in range(i + 1, len(setups)):
            setup1_label = setup_labels[i]
            setup2_label = setup_labels[j]

            # Calculate statistics for each benchmark
            per_benchmark_stats: dict[str, BenchmarkStats] = {}

            for benchmark_spec in benchmarks:
                benchmark_name, subset, limit = _parse_benchmark_spec(benchmark_spec)
                display_name = benchmark_spec

                setup1_results = all_setup_results[i][display_name]
                setup2_results = all_setup_results[j][display_name]

                # Calculate stats for this benchmark with limit
                bench_stats = _calculate_benchmark_stats(
                    display_name,
                    setup1_label,
                    setup2_label,
                    setup1_results,
                    setup2_results,
                    limit=limit,
                )

                if bench_stats:
                    per_benchmark_stats[display_name] = bench_stats

            # Calculate aggregate statistics across all benchmarks
            aggregate_stats = _calculate_aggregate_stats(setup1_label, setup2_label, list(per_benchmark_stats.values()))

            # Store pairwise comparison
            all_pairwise_comparisons.append(
                PairwiseComparison(
                    setup1_label=setup1_label,
                    setup2_label=setup2_label,
                    per_benchmark=per_benchmark_stats,
                    aggregate_stats=aggregate_stats,
                )
            )

    # Render results
    if output_format == "text":
        # Render per-benchmark tables
        for benchmark_spec in benchmarks:
            # Collect stats for this benchmark from all pairwise comparisons
            benchmark_stats_for_display = []
            for comparison in all_pairwise_comparisons:
                if benchmark_spec in comparison.per_benchmark:
                    benchmark_stats_for_display.append(comparison.per_benchmark[benchmark_spec])
            if benchmark_stats_for_display:
                _render_pairwise_summary_table(
                    benchmark_stats_for_display,
                    title=f"Comparison Results: {benchmark_spec}",
                )

        # Render overall table with Breslow-Day test (if multiple benchmarks)
        if len(benchmarks) > 1:
            # Render each pairwise comparison with its own Breslow-Day result
            aggregate_stats_list = []
            for comparison in all_pairwise_comparisons:
                if comparison.aggregate_stats:
                    aggregate_stats_list.append(comparison.aggregate_stats)

            _render_pairwise_summary_table(
                aggregate_stats_list,
                title="Overall Comparison (All Benchmarks)",
            )

    else:
        # JSON output
        output = {
            "setups": setup_labels,
            "per_benchmark": {},
            "overall": {"pairwise_comparisons": []},
        }

        # Add per-benchmark results from all pairwise comparisons
        for benchmark_spec in benchmarks:
            benchmark_results = []
            for comparison in all_pairwise_comparisons:
                if benchmark_spec in comparison.per_benchmark:
                    stats = comparison.per_benchmark[benchmark_spec]
                    benchmark_results.append(
                        {
                            "setup1": stats.setup1_label,
                            "setup2": stats.setup2_label,
                            "success_rate1": float(stats.success_rate1),
                            "success_rate2": float(stats.success_rate2),
                            "rate_difference": float(stats.rate_diff),
                            "p_value": float(stats.p_value),
                            "num_tasks": int(stats.num_tasks),
                            "is_significant": bool(stats.is_significant),
                            "winner": stats.winner,
                        }
                    )
            if benchmark_results:
                output["per_benchmark"][benchmark_spec] = benchmark_results

        # Add overall results with Breslow-Day
        for comparison in all_pairwise_comparisons:
            if comparison.aggregate_stats:
                stats = comparison.aggregate_stats
                # Get Breslow-Day result from comparison object
                if (
                    comparison.aggregate_stats.breslow_day_pvalue is not None
                    and comparison.aggregate_stats.breslow_day_interpretation is not None
                ):
                    breslow_day = {
                        "p_value": float(comparison.aggregate_stats.breslow_day_pvalue),
                        "interpretation": comparison.aggregate_stats.breslow_day_interpretation,
                    }
                else:
                    breslow_day = None

                output["overall"]["pairwise_comparisons"].append(
                    {
                        "setup1": stats.setup1_label,
                        "setup2": stats.setup2_label,
                        "success_rate1": float(stats.success_rate1),
                        "success_rate2": float(stats.success_rate2),
                        "rate_difference": float(stats.rate_diff),
                        "p_value": float(stats.p_value),
                        "num_tasks": int(stats.num_tasks),
                        "is_significant": bool(stats.is_significant),
                        "winner": stats.winner,
                        "breslow_day": breslow_day,
                    }
                )

        print(json.dumps(output, indent=2))


__all__ = ["compare_cmd"]

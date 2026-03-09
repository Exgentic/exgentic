# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich_click as click

BENCHMARK_MAP = {
    "appworld_test_normal": "AppWorld",
    "browsecompplus": "BrowseComp+",
    "swebench": "SWE-bench",
    "tau2_airline": "TauBench-Airline",
    "tau2_retail": "TauBench-Retail",
    "tau2_telecom": "TauBench-Telecom",
}

AGENT_MAP = {
    "claude_code": "claude-code",
    "openai_solo": "openai-mcp",
    "smolagents_code": "smolagents",
    "tool_calling": "litellm-react",
    "tool_calling_with_shortlisting": "litellm-shortlist",
}

MODEL_MAP = {
    "openai_aws_claude-opus-4-5": "claude-opus-4.5",
    "openai_Azure_gpt-5.2-2025-12-11": "gpt-5.2",
    "openai_gcp_gemini-3-pro-preview": "gemini-3-pro",
}

AGENT_SHAPES = {
    "claude-code": "*",
    "litellm-react": "o",
    "litellm-shortlist": "s",
    "openai-mcp": "D",
    "smolagents": "^",
}

MODEL_COLORS = {
    "claude-opus-4.5": "#E69F00",
    "gemini-3-pro": "#56B4E9",
    "gpt-5.2": "#009E73",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _coerce_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _normalize_results_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "benchmark" in out.columns:
        out["benchmark"] = out["benchmark"].map(BENCHMARK_MAP).fillna(out["benchmark"])
    else:
        raise click.ClickException("CSV missing required column: benchmark")

    if "agent" in out.columns:
        out["agent"] = out["agent"].map(AGENT_MAP).fillna(out["agent"])
    elif "agent_normalized" in out.columns:
        out["agent"] = out["agent_normalized"]
    else:
        raise click.ClickException("CSV missing required column: agent")

    if "model" in out.columns:
        out["model"] = out["model"].map(MODEL_MAP).fillna(out["model"])
    elif "model_normalized" in out.columns:
        out["model"] = out["model_normalized"]
    else:
        raise click.ClickException("CSV missing required column: model")

    if "agent_normalized" not in out.columns:
        out["agent_normalized"] = out["agent"]
    if "model_normalized" not in out.columns:
        out["model_normalized"] = out["model"]

    if "score" in out.columns:
        out["score"] = _coerce_numeric(out, "score")
    elif "benchmark_score" in out.columns:
        out["score"] = _coerce_numeric(out, "benchmark_score")
    else:
        raise click.ClickException("CSV missing benchmark_score (average_score is not allowed).")

    if "avg_steps" not in out.columns and "average_steps" in out.columns:
        out["avg_steps"] = _coerce_numeric(out, "average_steps")

    if "num_tasks" not in out.columns:
        if "total_sessions" in out.columns:
            out["num_tasks"] = _coerce_numeric(out, "total_sessions")
        elif "planned_sessions" in out.columns:
            out["num_tasks"] = _coerce_numeric(out, "planned_sessions")

    if "avg_cost" not in out.columns:
        if "average_run_cost" in out.columns:
            out["avg_cost"] = _coerce_numeric(out, "average_run_cost")
        elif "average_agent_cost" in out.columns:
            out["avg_cost"] = _coerce_numeric(out, "average_agent_cost")
        elif "total_run_cost" in out.columns and "total_sessions" in out.columns:
            total_cost = _coerce_numeric(out, "total_run_cost")
            total_sessions = _coerce_numeric(out, "total_sessions")
            out["avg_cost"] = total_cost / total_sessions.replace(0, np.nan)

    if "total_cost" not in out.columns and "total_run_cost" in out.columns:
        out["total_cost"] = _coerce_numeric(out, "total_run_cost")

    if "finished_pct" not in out.columns and "percent_finished" in out.columns:
        out["finished_pct"] = _coerce_numeric(out, "percent_finished")

    return out


def _get_benchmark_weight(benchmark: str) -> float:
    if "TauBench" in benchmark:
        return 1.0 / 3.0
    return 1.0


def _load_normalized_frames(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df = _normalize_results_df(df)
    valid_df = df[df["score"].notna()].copy()
    if valid_df.empty:
        raise click.ClickException("No valid scores found in CSV.")
    valid_df["benchmark_weight"] = valid_df["benchmark"].apply(_get_benchmark_weight)
    return df, valid_df


def _compute_weighted_scores(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    config_scores: dict[tuple[str, str], float] = {}
    for (agent, model), group in df.groupby(["agent_normalized", "model_normalized"]):
        scores = []
        weights = []
        for _, row in group.iterrows():
            if pd.notna(row["score"]):
                scores.append(row["score"])
                weights.append(_get_benchmark_weight(row["benchmark"]))
        if scores:
            config_scores[(agent, model)] = float(np.average(scores, weights=weights))
    return config_scores


def _compute_pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pareto: list[tuple[float, float]] = []
    for i, (x1, y1) in enumerate(points):
        is_pareto = True
        for j, (x2, y2) in enumerate(points):
            if i == j:
                continue
            if x2 <= x1 and y2 >= y1 and (x2 < x1 or y2 > y1):
                is_pareto = False
                break
        if is_pareto:
            pareto.append((x1, y1))
    pareto.sort(key=lambda p: p[0])
    return pareto


def _build_leaderboard_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    valid_df = df[df["score"].notna()].copy()
    valid_df["benchmark_weight"] = valid_df["benchmark"].apply(_get_benchmark_weight)

    pivot_scores = valid_df.pivot_table(
        index=["agent_normalized", "model_normalized"],
        columns="benchmark",
        values="score",
        aggfunc="mean",
    )

    config_metrics = []
    for (agent, model), group in valid_df.groupby(["agent_normalized", "model_normalized"]):
        mean_score = np.average(group["score"], weights=group["benchmark_weight"])

        if "avg_steps" in group.columns and group["avg_steps"].notna().any():
            mean_steps = np.average(
                group[group["avg_steps"].notna()]["avg_steps"],
                weights=group[group["avg_steps"].notna()]["benchmark_weight"],
            )
        else:
            mean_steps = np.nan

        if "avg_cost" in group.columns and group["avg_cost"].notna().any():
            mean_cost = np.average(
                group[group["avg_cost"].notna()]["avg_cost"],
                weights=group[group["avg_cost"].notna()]["benchmark_weight"],
            )
        else:
            mean_cost = np.nan

        config_metrics.append(
            {
                "agent": agent,
                "model": model,
                "mean_score": mean_score,
                "mean_steps": mean_steps,
                "mean_cost": mean_cost,
            }
        )

    config_df = pd.DataFrame(config_metrics)

    config_points = {}
    config_weighted_comparisons = {}
    for _bench, group in valid_df.groupby("benchmark"):
        weight = group["benchmark_weight"].iloc[0]
        config_scores = {}
        for _, row in group.iterrows():
            config = (row["agent_normalized"], row["model_normalized"])
            score = row["score"]
            if pd.notna(score):
                config_scores[config] = score
        configs = list(config_scores.keys())
        for i, c1 in enumerate(configs):
            for c2 in configs[i + 1 :]:
                s1 = config_scores[c1]
                s2 = config_scores[c2]
                if c1 not in config_points:
                    config_points[c1] = 0.0
                    config_weighted_comparisons[c1] = 0.0
                if c2 not in config_points:
                    config_points[c2] = 0.0
                    config_weighted_comparisons[c2] = 0.0
                if s1 > s2:
                    config_points[c1] += 1.0 * weight
                elif s1 == s2:
                    config_points[c1] += 0.5 * weight
                    config_points[c2] += 0.5 * weight
                else:
                    config_points[c2] += 1.0 * weight
                config_weighted_comparisons[c1] += weight
                config_weighted_comparisons[c2] += weight

    win_rates = {}
    for config in config_points:
        if config_weighted_comparisons[config] > 0:
            win_rates[config] = config_points[config] / config_weighted_comparisons[config]
        else:
            win_rates[config] = np.nan

    config_df["win_rate"] = config_df.apply(lambda row: win_rates.get((row["agent"], row["model"]), np.nan), axis=1)

    pivot_scores_reset = pivot_scores.reset_index()
    full_table = config_df.merge(
        pivot_scores_reset,
        left_on=["agent", "model"],
        right_on=["agent_normalized", "model_normalized"],
        how="left",
    ).sort_values("mean_score", ascending=False)

    benchmarks = list(pivot_scores.columns)
    return full_table, benchmarks


def _generate_leaderboard(df: pd.DataFrame) -> str:
    full_table, benchmarks = _build_leaderboard_table(df)
    benchmark_short_names = {
        "SWE-bench": "SWE",
        "BrowseComp+": "Browse",
        "TauBench-Airline": "Airline",
        "TauBench-Retail": "Retail",
        "TauBench-Telecom": "Telecom",
        "AppWorld": "App",
    }
    bench_display = [benchmark_short_names.get(b, b) for b in benchmarks]

    agent_short = {
        "litellm-react": "React",
        "litellm-shortlist": "React+Short",
        "smolagents": "Smol",
        "openai-mcp": "OpenAI-MCP",
        "claude-code": "Claude-Code",
    }
    model_short = {
        "gpt-5.2": "GPT-5.2",
        "claude-opus-4.5": "Opus-4.5",
        "gemini-3-pro": "Gemini-3",
    }

    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\caption{Agent-Model Configuration Leaderboard}")
    latex.append(r"\label{tab:leaderboard}")

    num_benchmarks = len(benchmarks)
    col_spec = "ll" + "c" * num_benchmarks + "cccc"
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")

    header1 = r"\textbf{Agent} & \textbf{Model}"
    for bench_name in bench_display:
        header1 += f" & \\textbf{{{bench_name}}}"
    header1 += r" & \textbf{Mean} & \textbf{Win} & \textbf{Steps} & \textbf{Cost} \\"
    latex.append(header1)

    header2 = r" & "
    for _ in benchmarks:
        header2 += " & "
    header2 += r" & Score & Rate & (avg) & (\$) \\"
    latex.append(header2)
    latex.append(r"\midrule")

    for _, row in full_table.iterrows():
        agent_name = agent_short.get(row["agent"], row["agent"])
        model_name = model_short.get(row["model"], row["model"])
        row_str = f"{agent_name} & {model_name}"
        for bench in benchmarks:
            score = row.get(bench, np.nan)
            if pd.notna(score):
                row_str += f" & {score:.2f}"
            else:
                row_str += " & --"
        if pd.notna(row["mean_score"]):
            row_str += f" & {row['mean_score']:.2f}"
        else:
            row_str += " & --"
        if pd.notna(row["win_rate"]):
            row_str += f" & {row['win_rate']:.2f}"
        else:
            row_str += " & --"
        if pd.notna(row["mean_steps"]):
            row_str += f" & {row['mean_steps']:.1f}"
        else:
            row_str += " & --"
        if pd.notna(row["mean_cost"]):
            row_str += f" & {row['mean_cost']:.2f}"
        else:
            row_str += " & --"
        row_str += r" \\"
        latex.append(row_str)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


@click.group("analyse")
def analyse_cmd() -> None:
    """Analyze result CSVs without intermediate files."""


@analyse_cmd.command("leaderboard")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyse_leaderboard_cmd(csv_path: Path) -> None:
    """Generate leaderboard table from a results CSV."""
    df = pd.read_csv(csv_path)
    df = _normalize_results_df(df)
    click.echo(_generate_leaderboard(df))


@analyse_cmd.command("leaderboard-paper")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path for the paper-style leaderboard table.",
)
def analyse_leaderboard_paper_cmd(csv_path: Path, output_path: Path | None) -> None:
    """Generate the paper-style leaderboard table and save to file."""
    df = pd.read_csv(csv_path)
    df = _normalize_results_df(df)
    full_table, benchmarks = _build_leaderboard_table(df)

    bench_order = [
        "AppWorld",
        "BrowseComp+",
        "SWE-bench",
        "TauBench-Airline",
        "TauBench-Retail",
        "TauBench-Telecom",
    ]
    bench_map = {b: b for b in benchmarks}
    ordered_benchmarks = [b for b in bench_order if b in bench_map]

    agent_macro = {
        "openai-mcp": r"\solo{}",
        "smolagents": r"\smol{}",
        "litellm-react": r"\react{}",
        "litellm-shortlist": r"\short{}",
        "claude-code": r"\cc{}",
    }
    model_macro = {
        "claude-opus-4.5": r"\opus{}",
        "gemini-3-pro": r"\gemini{}",
        "gpt-5.2": r"\gpt{}",
    }

    header = r"""\definecolor{tableheader}{RGB}{248, 249, 250}
\definecolor{rowgray}{RGB}{252, 252, 253}
\definecolor{benchmarkbg}{RGB}{245, 245, 247}
% TABLE CODE (put this where you want the table):
\begin{table}[t!]
\centering
\begin{tcolorbox}[
    colback=white,
    colframe=gray!20,
    boxrule=0.5pt,
    arc=3pt,
    outer arc=3pt,
    width=\columnwidth,
    top=0pt,
    bottom=0pt,
    left=-1pt,
    right=1.5pt,
    boxsep=0pt
]
\resizebox{0.8\textwidth}{!}{%
\renewcommand{\arraystretch}{1.7}
\setlength{\tabcolsep}{3pt}
\footnotesize
\begin{tabular}{@{}l l c c c !{\color{gray!20}\vrule} >{\columncolor{benchmarkbg}}c """
    r""">{\columncolor{benchmarkbg}}c >{\columncolor{benchmarkbg}}c >{\columncolor{benchmarkbg}}c """
    r""">{\columncolor{benchmarkbg}}c >{\columncolor{benchmarkbg}}c@{}}
\rowcolor{tableheader}
\textbf{\#} & \textbf{\shortstack{General Agent}} & \scriptsize\textbf{Model} & """
    r"""\shortstack{\scriptsize{Avg}\\\textbf{Success}} & \shortstack{\scriptsize{Avg}\\\textbf{Cost}} & """
    r"""\cellcolor{tableheader}\tiny\textbf{\shortstack{App\\World}} & """
    r"""\cellcolor{tableheader}\tiny\textbf{\shortstack{Browse\\Comp+}} & """
    r"""\cellcolor{tableheader}\tiny\textbf{\shortstack{SWE\\benchV}} & """
    r"""\cellcolor{tableheader}\tiny\textbf{\shortstack{Tau 2\\Airline}} & """
    r"""\cellcolor{tableheader}\tiny\textbf{\shortstack{Tau 2\\Retail}} & """
    r"""\cellcolor{tableheader}\tiny\textbf{\shortstack{Tau 2\\Telecom}} \\
"""

    def fmt_score(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return "--"
        text = f"{value:.2f}"
        if text.startswith("0"):
            text = text[1:]
        return text

    def fmt_cost(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return "--"
        return f"${value:.1f}"

    rows = []
    for idx, (_, row) in enumerate(full_table.iterrows(), start=1):
        agent = row["agent"]
        model = row["model"]
        mean_score = row["mean_score"]
        mean_cost = row["mean_cost"]
        agent_cell = agent_macro.get(agent, agent)
        model_cell = model_macro.get(model, model)
        prefix = r"\rowcolor{rowgray}" if idx % 2 == 0 else ""
        cells = [
            str(idx),
            agent_cell,
            rf"{{\scriptsize {model_cell}}}",
            fmt_score(mean_score),
            fmt_cost(mean_cost),
        ]
        for bench in ordered_benchmarks:
            score = row.get(bench, np.nan)
            score_cell = f"{{\\scriptsize {fmt_score(score)}}}" if pd.notna(score) else "{\\scriptsize --}"
            if prefix:
                score_cell = rf"\cellcolor{{rowgray}}{score_cell}"
            cells.append(score_cell)
        row_line = prefix + " " + " & ".join(cells) + r" \\"
        rows.append(row_line)

    footer = r"""
\end{tabular}%
}
\end{tcolorbox}
\caption{The \leaderboard{} comparing emerging general agents across standardized benchmarks.
Average Success represents the mean success rate across benchmarks; Average Cost represents the mean cost per task.
Performance is strongly influenced by backbone model choice.}
\label{tab:leaderboard}
\end{table}
"""

    latex = header + "\n".join(rows) + footer
    if output_path is None:
        output_path = _project_root() / "misc" / "paper" / "tables" / "leaderboard_paper.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    click.echo(f"Saved to {output_path}")


@analyse_cmd.command("model-agent")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyse_model_agent_cmd(csv_path: Path) -> None:
    """Model vs agent variance, pair means, and interaction analysis."""
    _, valid_df = _load_normalized_frames(csv_path)

    def weighted_mean(group):
        return np.average(group["score"], weights=group["benchmark_weight"])

    def weighted_std(group):
        wmean = np.average(group["score"], weights=group["benchmark_weight"])
        variance = np.average((group["score"] - wmean) ** 2, weights=group["benchmark_weight"])
        return np.sqrt(variance)

    agent_means = valid_df.groupby("agent_normalized").apply(weighted_mean)
    model_means = valid_df.groupby("model_normalized").apply(weighted_mean)
    valid_df["agent_mean"] = valid_df["agent_normalized"].map(agent_means)
    valid_df["model_mean"] = valid_df["model_normalized"].map(model_means)

    grand_mean = np.average(valid_df["score"], weights=valid_df["benchmark_weight"])
    total_var = np.average((valid_df["score"] - grand_mean) ** 2, weights=valid_df["benchmark_weight"])
    agent_var = np.average((valid_df["agent_mean"] - grand_mean) ** 2, weights=valid_df["benchmark_weight"])
    model_var = np.average((valid_df["model_mean"] - grand_mean) ** 2, weights=valid_df["benchmark_weight"])

    click.echo("MODEL VS AGENT (benchmark-weighted)")
    click.echo(f"Total variance: {total_var:.4f}")
    click.echo(f"Agent variance: {agent_var:.4f} ({100*agent_var/total_var:.1f}%)")
    click.echo(f"Model variance: {model_var:.4f} ({100*model_var/total_var:.1f}%)")

    click.echo("\nBy Model (weighted):")
    by_model = (
        valid_df.groupby("model_normalized")
        .apply(lambda g: pd.Series({"mean": weighted_mean(g), "std": weighted_std(g), "count": len(g)}))
        .round(3)
    )
    click.echo(by_model.to_string())

    click.echo("\nBy Agent (weighted):")
    by_agent = (
        valid_df.groupby("agent_normalized")
        .apply(lambda g: pd.Series({"mean": weighted_mean(g), "std": weighted_std(g), "count": len(g)}))
        .round(3)
    )
    click.echo(by_agent.to_string())

    click.echo("\nAgent-Model pair weighted means:")
    pair_means = valid_df.groupby(["agent_normalized", "model_normalized"]).apply(weighted_mean).round(3)
    for (agent, model), score in pair_means.items():
        click.echo(f"{agent:<20} {model:<20} {score:>10.3f}")

    click.echo("\nInteraction analysis (cell means):")
    cell_means = valid_df.groupby(["model_normalized", "agent_normalized"])["score"].mean()
    df_cells = cell_means.reset_index()
    grand_mean = df_cells["score"].mean()
    model_effects = df_cells.groupby("model_normalized")["score"].mean() - grand_mean
    agent_effects = df_cells.groupby("agent_normalized")["score"].mean() - grand_mean

    interactions = []
    for _, row in df_cells.iterrows():
        m = row["model_normalized"]
        a = row["agent_normalized"]
        score = row["score"]
        pred = grand_mean + model_effects[m] + agent_effects[a]
        interactions.append(score - pred)

    var_model = model_effects.var()
    var_agent = agent_effects.var()
    var_interact = np.var(interactions)
    total_comp = var_model + var_agent + var_interact
    click.echo(f"Model Main Effect:  {100*var_model/total_comp:.1f}%")
    click.echo(f"Agent Main Effect:  {100*var_agent/total_comp:.1f}%")
    click.echo(f"Interaction Effect:{100*var_interact/total_comp:.1f}%")


@analyse_cmd.command("model-win-rate")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyse_model_win_rate_cmd(csv_path: Path) -> None:
    """Model win rate (TauBench weighted)."""
    _, valid_df = _load_normalized_frames(csv_path)
    model_points = {m: 0.0 for m in valid_df["model_normalized"].unique()}
    model_weighted_comparisons = {m: 0.0 for m in model_points}

    for (_agent, _bench), group in valid_df.groupby(["agent_normalized", "benchmark"]):
        weight = group["benchmark_weight"].iloc[0]
        model_scores = {}
        for _, row in group.iterrows():
            model = row["model_normalized"]
            score = row["score"]
            if pd.notna(score):
                model_scores[model] = score
        models = list(model_scores.keys())
        for i, m1 in enumerate(models):
            for m2 in models[i + 1 :]:
                s1 = model_scores[m1]
                s2 = model_scores[m2]
                if s1 > s2:
                    model_points[m1] += 1.0 * weight
                elif s1 == s2:
                    model_points[m1] += 0.5 * weight
                    model_points[m2] += 0.5 * weight
                else:
                    model_points[m2] += 1.0 * weight
                model_weighted_comparisons[m1] += weight
                model_weighted_comparisons[m2] += weight

    click.echo(f"{'Model':<20} {'Win Rate':>10} {'Weighted Comparisons':>20}")
    click.echo("-" * 55)
    for model in sorted(model_points.keys()):
        if model_weighted_comparisons[model] > 0:
            win_rate = model_points[model] / model_weighted_comparisons[model]
            click.echo(f"{model:<20} {100*win_rate:>9.1f}% {model_weighted_comparisons[model]:>20.1f}")


@analyse_cmd.command("config-win-rate")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyse_config_win_rate_cmd(csv_path: Path) -> None:
    """Top configuration win rates (TauBench weighted)."""
    _, valid_df = _load_normalized_frames(csv_path)
    config_points = {}
    config_weighted_comparisons = {}

    for _bench, group in valid_df.groupby("benchmark"):
        weight = group["benchmark_weight"].iloc[0]
        config_scores = {}
        for _, row in group.iterrows():
            config = (row["agent_normalized"], row["model_normalized"])
            score = row["score"]
            if pd.notna(score):
                config_scores[config] = score
        configs = list(config_scores.keys())
        for i, c1 in enumerate(configs):
            for c2 in configs[i + 1 :]:
                s1 = config_scores[c1]
                s2 = config_scores[c2]
                if c1 not in config_points:
                    config_points[c1] = 0.0
                    config_weighted_comparisons[c1] = 0.0
                if c2 not in config_points:
                    config_points[c2] = 0.0
                    config_weighted_comparisons[c2] = 0.0
                if s1 > s2:
                    config_points[c1] += 1.0 * weight
                elif s1 == s2:
                    config_points[c1] += 0.5 * weight
                    config_points[c2] += 0.5 * weight
                else:
                    config_points[c2] += 1.0 * weight
                config_weighted_comparisons[c1] += weight
                config_weighted_comparisons[c2] += weight

    config_win_rates = []
    for config in config_points:
        if config_weighted_comparisons[config] > 0:
            win_rate = config_points[config] / config_weighted_comparisons[config]
            config_win_rates.append((config, win_rate, config_weighted_comparisons[config]))
    config_win_rates.sort(key=lambda x: x[1], reverse=True)

    click.echo(f"{'Agent':<20} {'Model':<20} {'Win Rate':>10} {'Weighted Comps':>15}")
    click.echo("-" * 70)
    for (agent, model), win_rate, comps in config_win_rates[:10]:
        click.echo(f"{agent:<20} {model:<20} {100*win_rate:>9.1f}% {comps:>15.1f}")


@analyse_cmd.command("best-per-benchmark")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyse_best_per_benchmark_cmd(csv_path: Path) -> None:
    """Best configuration per benchmark (top 3)."""
    _, valid_df = _load_normalized_frames(csv_path)
    for bench in sorted(valid_df["benchmark"].unique()):
        bench_df = valid_df[valid_df["benchmark"] == bench]
        best = bench_df.loc[bench_df["score"].idxmax()]
        click.echo(f"\n{bench}:")
        click.echo(f"  Winner: {best['agent_normalized']} + {best['model_normalized']}")
        click.echo(f"  Score: {best['score']:.3f}")
        top3 = bench_df.nlargest(3, "score")[["agent_normalized", "model_normalized", "score"]]
        click.echo("  Top 3:")
        for _, row in top3.iterrows():
            click.echo(f"    {row['agent_normalized']:20s} + {row['model_normalized']:20s} = {row['score']:.3f}")


@analyse_cmd.command("tool-shortlist")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyse_tool_shortlist_cmd(csv_path: Path) -> None:
    """Tool shortlisting effect (AppWorld)."""
    df, _ = _load_normalized_frames(csv_path)
    appworld = df[df["benchmark"] == "AppWorld"].copy()
    click.echo(f"Total AppWorld rows: {len(appworld)}")
    for model in ["gpt-5.2", "claude-opus-4.5", "gemini-3-pro"]:
        click.echo(f"\n{model}:")
        no_shortlist = appworld[
            (appworld["model_normalized"] == model) & (appworld["agent_normalized"] == "litellm-react")
        ]
        with_shortlist = appworld[
            (appworld["model_normalized"] == model) & (appworld["agent_normalized"] == "litellm-shortlist")
        ]
        if len(no_shortlist) > 0:
            score_no = no_shortlist["score"].values[0]
            click.echo(f"  Without shortlist: {score_no:.3f}")
        else:
            click.echo("  Without shortlist: NO DATA")
            score_no = None
        if len(with_shortlist) > 0:
            score_with = with_shortlist["score"].values[0]
            click.echo(f"  With shortlist:    {score_with:.3f}")
            if score_no is not None:
                click.echo(f"  Delta:             {score_with - score_no:+.3f}")
        else:
            click.echo("  With shortlist:    NO DATA")


@analyse_cmd.command("cost-efficiency")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyse_cost_efficiency_cmd(csv_path: Path) -> None:
    """Cost-efficiency analysis (benchmark-weighted)."""
    _, valid_df = _load_normalized_frames(csv_path)
    if "avg_cost" not in valid_df.columns or valid_df["avg_cost"].notna().sum() == 0:
        raise click.ClickException("avg_cost is required for cost-efficiency analysis.")
    cost_df = valid_df[valid_df["avg_cost"].notna()].copy()
    cost_df["efficiency"] = cost_df["score"] / cost_df["avg_cost"]

    by_config = (
        cost_df.groupby(["agent_normalized", "model_normalized"])
        .apply(
            lambda g: pd.Series(
                {
                    "score": np.average(g["score"], weights=g["benchmark_weight"]),
                    "avg_cost": np.average(g["avg_cost"], weights=g["benchmark_weight"]),
                }
            )
        )
        .reset_index()
    )
    by_config["efficiency"] = by_config["score"] / by_config["avg_cost"]
    top_efficient = by_config.nlargest(10, "efficiency")
    click.echo(f"{'Agent':<20} {'Model':<20} {'Score':>8} {'Cost':>10} {'Efficiency':>12}")
    click.echo("-" * 75)
    for _, row in top_efficient.iterrows():
        click.echo(
            f"{row['agent_normalized']:<20} {row['model_normalized']:<20} "
            f"{row['score']:>8.3f} ${row['avg_cost']:>9.2f} {row['efficiency']:>12.2f}"
        )


@analyse_cmd.command("component-impact")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyse_component_impact_cmd(csv_path: Path) -> None:
    """Component impact analysis."""
    _, valid_df = _load_normalized_frames(csv_path)
    agent_components = {
        "litellm-react": {
            "runtime": False,
            "shortlist": False,
            "schema_guard": False,
            "memory": False,
            "planning": False,
        },
        "litellm-shortlist": {
            "runtime": False,
            "shortlist": True,
            "schema_guard": False,
            "memory": False,
            "planning": False,
        },
        "smolagents": {
            "runtime": True,
            "shortlist": False,
            "schema_guard": True,
            "memory": False,
            "planning": False,
        },
        "openai-mcp": {
            "runtime": False,
            "shortlist": False,
            "schema_guard": True,
            "memory": False,
            "planning": False,
        },
        "claude-code": {
            "runtime": True,
            "shortlist": False,
            "schema_guard": True,
            "memory": True,
            "planning": True,
        },
    }

    for comp in ["runtime", "shortlist", "schema_guard", "memory", "planning"]:
        valid_df[comp] = valid_df["agent_normalized"].apply(lambda a, c=comp: agent_components.get(a, {}).get(c, False))

    click.echo(f"{'Component':<15} {'With':>8} {'Without':>8} {'Delta':>8} {'N_with':>8} {'N_without':>10}")
    click.echo("-" * 70)
    for comp in ["runtime", "shortlist", "schema_guard", "memory", "planning"]:
        with_comp_df = valid_df[valid_df[comp]]
        without_comp_df = valid_df[~valid_df[comp]]
        if len(with_comp_df) > 0 and len(without_comp_df) > 0:
            mean_with = np.average(with_comp_df["score"], weights=with_comp_df["benchmark_weight"])
            mean_without = np.average(without_comp_df["score"], weights=without_comp_df["benchmark_weight"])
            delta = mean_with - mean_without
            click.echo(
                f"{comp.replace('_', ' ').title():<15} {mean_with:>8.3f} {mean_without:>8.3f} "
                f"{delta:>+8.3f} {len(with_comp_df):>8} {len(without_comp_df):>10}"
            )


@analyse_cmd.command("correlation")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyse_correlation_cmd(csv_path: Path) -> None:
    """Cross-benchmark rank correlation."""
    _, valid_df = _load_normalized_frames(csv_path)
    pivot = valid_df.pivot_table(
        index=["agent_normalized", "model_normalized"],
        columns="benchmark",
        values="score",
    )
    if pivot.shape[1] < 2:
        raise click.ClickException("Need at least 2 benchmarks for correlation.")
    corr = pivot.corr(method="spearman")
    click.echo("Spearman Rank Correlation Matrix:")
    click.echo(corr.round(2).to_string())
    click.echo("\nNotable correlations (|r| > 0.7):")
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) > 0.7:
                click.echo(f"  {corr.columns[i]} vs {corr.columns[j]}: {val:.2f}")


@analyse_cmd.command("cost-score")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output PDF path for the cost-performance plot.",
)
def analyse_cost_score_cmd(csv_path: Path, output_path: Path | None) -> None:
    """Generate cost vs score plot with Pareto frontier."""
    df, _ = _load_normalized_frames(csv_path)
    df["benchmark_weight"] = df["benchmark"].apply(_get_benchmark_weight)

    weighted_scores = _compute_weighted_scores(df)
    if "avg_cost" not in df.columns or df["avg_cost"].notna().sum() == 0:
        raise click.ClickException("avg_cost is required for cost-score plot.")

    cost_df = df[df["avg_cost"].notna()].copy()
    cost_by_config = cost_df.groupby(["agent_normalized", "model_normalized"]).agg({"avg_cost": "mean"}).reset_index()

    agent_display_names = {
        "claude-code": "Claude Code",
        "litellm-react": "ReAct",
        "litellm-shortlist": "ReAct Short",
        "openai-mcp": "Solo",
        "smolagents": "Smolagent",
    }
    model_display_names = {
        "claude-opus-4.5": "Opus",
        "gemini-3-pro": "Gemini",
        "gpt-5.2": "GPT",
    }

    plot_data = []
    for _, row in cost_by_config.iterrows():
        agent = row["agent_normalized"]
        model = row["model_normalized"]
        config = (agent, model)
        if config in weighted_scores:
            plot_data.append(
                {
                    "agent": agent,
                    "model": model,
                    "cost": row["avg_cost"],
                    "score": weighted_scores[config],
                    "label": f"{agent_display_names.get(agent, agent)}\n{model_display_names.get(model, model)}",
                }
            )

    if not plot_data:
        raise click.ClickException("No plot data available.")

    plot_df = pd.DataFrame(plot_data)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    for _idx, (_, row) in enumerate(plot_df.iterrows()):
        shape = AGENT_SHAPES[row["agent"]]
        color = MODEL_COLORS[row["model"]]
        marker_size = 260 if shape == "*" else 200
        ax.scatter(
            row["cost"],
            row["score"],
            marker=shape,
            s=marker_size,
            c=color,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            zorder=3,
        )

    # Labels intentionally disabled.

    points = list(zip(plot_df["cost"].tolist(), plot_df["score"].tolist()))
    pareto = _compute_pareto_frontier(points)
    if pareto:
        xs, ys = zip(*pareto)
        ax.plot(xs, ys, linestyle="--", color="gray", linewidth=1.5, zorder=2)

    ax.set_xlabel("Average Cost per Task ($)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    y_min = max(0, plot_df["score"].min() - 0.05)
    y_max = min(1.0, plot_df["score"].max() + 0.08)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.8, color="gray", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Legends
    agent_labels = {
        "claude-code": "Claude Code",
        "litellm-react": "ReAct",
        "litellm-shortlist": "ReAct Short",
        "openai-mcp": "OpenAI Solo",
        "smolagents": "Smolagent",
    }
    model_labels = {
        "claude-opus-4.5": "Opus 4.5",
        "gemini-3-pro": "Gemini 3",
        "gpt-5.2": "GPT 5.2",
    }

    agent_handles = []
    agent_label_list = []
    for agent in AGENT_SHAPES:
        agent_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=AGENT_SHAPES[agent],
                color="w",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=8,
                linestyle="",
            )
        )
        agent_label_list.append(agent_labels[agent])

    model_handles = []
    model_label_list = []
    for model in MODEL_COLORS:
        model_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor="black",
                markersize=8,
                linestyle="",
            )
        )
        model_label_list.append(model_labels[model])

    legend1 = ax.legend(
        model_handles,
        model_label_list,
        loc="lower right",
        bbox_to_anchor=(0.70, 0.05),
        frameon=False,
        fontsize=8,
        title="Model",
        title_fontsize=9,
    )
    ax.add_artist(legend1)

    ax.legend(
        agent_handles,
        agent_label_list,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.05),
        frameon=False,
        fontsize=8,
        title="Agent",
        title_fontsize=9,
    )

    if output_path is None:
        output_path = _project_root() / "misc" / "paper" / "figures" / "cost_performance.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    click.echo(f"Saved to {output_path}")
    click.echo(f"Saved to {output_path.with_suffix('.png')}")


def _save_pdf_png(fig: Any, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


@analyse_cmd.command("paper-figures")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--outdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory for generated figures.",
)
def analyse_paper_figures_cmd(csv_path: Path, outdir: Path | None) -> None:
    """Generate the paper graph set from one CSV."""
    df, valid_df = _load_normalized_frames(csv_path)
    if outdir is None:
        outdir = _project_root() / "misc" / "paper" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Cost-performance (reuse existing command output default file name).
    analyse_cost_score_cmd.callback(csv_path=csv_path, output_path=outdir / "cost_performance.pdf")  # type: ignore[attr-defined]

    # 2) Results heatmap (config x benchmark).
    heat = valid_df.pivot_table(
        index=["agent_normalized", "model_normalized"],
        columns="benchmark",
        values="score",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ylabels = [f"{a} | {m}" for a, m in heat.index]
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title("Results Heatmap")
    fig.colorbar(im, ax=ax, label="Score")
    _save_pdf_png(fig, outdir / "results_heatmap.pdf")
    click.echo(f"Saved to {outdir / 'results_heatmap.pdf'}")
    click.echo(f"Saved to {(outdir / 'results_heatmap.pdf').with_suffix('.png')}")

    # 3) Benchmark correlation heatmap (Spearman).
    corr = heat.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Benchmark Correlation (Spearman)")
    fig.colorbar(im, ax=ax, label="Correlation")
    _save_pdf_png(fig, outdir / "benchmark_correlation.pdf")
    click.echo(f"Saved to {outdir / 'benchmark_correlation.pdf'}")
    click.echo(f"Saved to {(outdir / 'benchmark_correlation.pdf').with_suffix('.png')}")

    # 4) Protocol comparison.
    protocol_map = {
        "litellm-react": "Tool-calling",
        "litellm-shortlist": "Tool-calling",
        "smolagents": "Python-functions",
        "openai-mcp": "MCP",
        "claude-code": "MCP",
    }
    by_protocol = valid_df.assign(protocol=valid_df["agent_normalized"].map(protocol_map)).dropna(subset=["protocol"])
    agg = (
        by_protocol.groupby(["benchmark", "protocol"])["score"]
        .mean()
        .reset_index()
        .pivot(index="benchmark", columns="protocol", values="score")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    agg.plot(kind="bar", ax=ax, rot=25)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Score")
    ax.set_title("Protocol Comparison by Benchmark")
    ax.legend(frameon=False)
    _save_pdf_png(fig, outdir / "protocol_comparison.pdf")
    click.echo(f"Saved to {outdir / 'protocol_comparison.pdf'}")
    click.echo(f"Saved to {(outdir / 'protocol_comparison.pdf').with_suffix('.png')}")

    # 5) Component impact.
    agent_components = {
        "litellm-react": {
            "runtime": False,
            "shortlist": False,
            "schema_guard": False,
            "memory": False,
            "planning": False,
        },
        "litellm-shortlist": {
            "runtime": False,
            "shortlist": True,
            "schema_guard": False,
            "memory": False,
            "planning": False,
        },
        "smolagents": {
            "runtime": True,
            "shortlist": False,
            "schema_guard": True,
            "memory": False,
            "planning": False,
        },
        "openai-mcp": {
            "runtime": False,
            "shortlist": False,
            "schema_guard": True,
            "memory": False,
            "planning": False,
        },
        "claude-code": {
            "runtime": True,
            "shortlist": False,
            "schema_guard": True,
            "memory": True,
            "planning": True,
        },
    }
    tmp = valid_df.copy()
    components = ["runtime", "shortlist", "schema_guard", "memory", "planning"]
    deltas = []
    for comp in components:
        tmp[comp] = tmp["agent_normalized"].apply(lambda a, c=comp: agent_components.get(a, {}).get(c, False))
        with_comp = tmp[tmp[comp]]["score"]
        without_comp = tmp[~tmp[comp]]["score"]
        if len(with_comp) and len(without_comp):
            deltas.append((comp, with_comp.mean() - without_comp.mean()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [c for c, _ in deltas]
    values = [v for _, v in deltas]
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Score Delta (With - Without)")
    ax.set_title("Component Impact")
    _save_pdf_png(fig, outdir / "component_impact.pdf")
    click.echo(f"Saved to {outdir / 'component_impact.pdf'}")
    click.echo(f"Saved to {(outdir / 'component_impact.pdf').with_suffix('.png')}")

    # 6) Best configuration per benchmark.
    winners = (
        valid_df.sort_values("score", ascending=False)
        .groupby("benchmark", as_index=False)
        .first()
        .sort_values("score", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [
        f"{b}\n{a} + {m}"
        for b, a, m in zip(
            winners["benchmark"],
            winners["agent_normalized"],
            winners["model_normalized"],
        )
    ]
    ax.barh(labels, winners["score"], color="#457b9d")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Best Per Benchmark")
    _save_pdf_png(fig, outdir / "best_per_benchmark.pdf")
    click.echo(f"Saved to {outdir / 'best_per_benchmark.pdf'}")
    click.echo(f"Saved to {(outdir / 'best_per_benchmark.pdf').with_suffix('.png')}")


__all__ = [
    "analyse_cmd",
    "analyse_leaderboard_cmd",
    "analyse_leaderboard_paper_cmd",
    "analyse_model_agent_cmd",
    "analyse_model_win_rate_cmd",
    "analyse_config_win_rate_cmd",
    "analyse_best_per_benchmark_cmd",
    "analyse_tool_shortlist_cmd",
    "analyse_cost_efficiency_cmd",
    "analyse_component_impact_cmd",
    "analyse_correlation_cmd",
    "analyse_cost_score_cmd",
    "analyse_paper_figures_cmd",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from .views import (
    RunState,
    build_history_tab,
    build_leaderboard_tab,
    build_run_tab,
    refresh_ui,
)

ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"


def create_ui() -> None:
    state = RunState()

    ui.page_title("Exgentic Dashboard")
    ui.add_head_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap"
              rel="stylesheet">
        <style>
            :root {
                font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
                --ui-bg: #f5f5f7;
                --ui-card: #ffffff;
                --ui-border: #e5e7eb;
                --ui-text: #111111;
                --ui-muted: #6b7280;
                --ui-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
                --q-primary: #111111;
                --q-secondary: #111111;
                --q-accent: #111111;
            }
            body, .q-app {
                font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
            }
            body, .q-app, .q-layout, .q-page, .q-page-container {
                background-color: var(--ui-bg) !important;
                color: var(--ui-text) !important;
            }
            .card {
                background: var(--ui-card);
                border: 1px solid var(--ui-border);
                border-radius: 18px;
                box-shadow: var(--ui-shadow);
            }
            .section-title {
                font-size: 1.1rem;
                font-weight: 600;
                letter-spacing: -0.01em;
            }
            .metric {
                font-weight: 500;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 1rem;
            }
            .metric-grid-4 {
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }
            .metric-card {
                background: var(--ui-card);
                border: 1px solid var(--ui-border);
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
                position: relative;
            }
            .metric-card-sm {
                border-radius: 12px;
                padding: 0.65rem 0.9rem;
                box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
            }
            .metric-card-sm .metric-label {
                font-size: 0.7rem;
            }
            .metric-card-sm .metric-value {
                font-size: 1.1rem;
                margin-top: 0.2rem;
            }
            .metric-card-lg {
                padding: 1.35rem 1.5rem;
                border-radius: 20px;
            }
            .metric-card-lg .metric-label {
                font-size: 0.8rem;
            }
            .metric-card-lg .metric-value {
                font-size: 2rem;
            }
            .metric-info {
                position: absolute;
                top: 0.65rem;
                right: 0.65rem;
                font-size: 0.9rem;
                color: var(--ui-muted);
                cursor: help;
            }
            .metric-card-sm .metric-info {
                top: 0.45rem;
                right: 0.45rem;
                font-size: 0.75rem;
            }
            .metric-card-lg .metric-info {
                top: 0.8rem;
                right: 0.8rem;
                font-size: 1rem;
            }
            .metric-label {
                color: var(--ui-muted);
                font-size: 0.85rem;
                letter-spacing: 0.02em;
                text-transform: uppercase;
            }
            .text-primary, .text-secondary, .text-accent {
                color: var(--ui-text) !important;
            }
            .bg-primary, .bg-secondary, .bg-accent {
                background: var(--ui-text) !important;
            }
            a, .q-link {
                color: var(--ui-text) !important;
            }
            .q-tab__indicator {
                background: var(--ui-text) !important;
            }
            .q-tab--active .q-tab__label,
            .q-tab--active .q-tab__icon {
                color: var(--ui-text) !important;
            }
            .q-btn--flat .q-btn__content,
            .q-btn--outline .q-btn__content,
            .q-btn--flat .q-icon,
            .q-btn--outline .q-icon {
                color: var(--ui-text) !important;
            }
            .q-field--focused .q-field__control:before,
            .q-field--focused .q-field__control:after {
                border-color: var(--ui-text) !important;
            }
            .q-field--focused .q-field__label {
                color: var(--ui-text) !important;
            }
            .q-checkbox__inner--truthy .q-checkbox__bg,
            .q-radio__inner--truthy .q-radio__bg,
            .q-toggle__inner--truthy .q-toggle__track {
                background: var(--ui-text) !important;
            }
            .q-checkbox__inner--truthy .q-checkbox__icon,
            .q-radio__inner--truthy .q-radio__icon {
                color: #ffffff !important;
            }
            .metric-value {
                font-size: 1.8rem;
                font-weight: 600;
                margin-top: 0.35rem;
            }
            .trajectory-box {
                border: 1px solid var(--ui-border);
                border-radius: 12px;
                padding: 0.75rem 1rem;
                background: #fafafa;
            }
            .trajectory-title {
                font-weight: 600;
                margin-bottom: 0.4rem;
            }
            .muted {
                color: var(--ui-muted);
            }
            .form-stack {
                gap: 0.75rem;
            }
            .split-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 1rem;
            }
            .q-field__control {
                border-radius: 12px;
            }
            .app-logo {
                height: 52px;
                width: 240px;
                max-width: 240px;
                object-fit: contain;
                display: block;
            }
            .app-logo img {
                object-fit: contain !important;
            }
            .start-run-btn {
                background: #39ff14 !important;
                color: #0b0f10 !important;
                font-weight: 600;
            }
            .start-run-btn:hover {
                filter: brightness(0.95);
            }
        </style>
        """
    )
    dark = ui.dark_mode()
    dark.disable()

    with ui.row().classes("w-full items-center justify-between"):
        logo_path = ASSETS_DIR.parent.parent / "misc" / "assets" / "exgentic_banner_black_no_background.png"
        if logo_path.is_file():
            ui.image(logo_path).classes("app-logo").props("fit=contain")
        with ui.tabs() as main_tabs:
            run_tab = ui.tab("Run")
            leaderboard_tab = ui.tab("Leaderboard")
            history_tab = ui.tab("History")

    with ui.tab_panels(main_tabs, value=run_tab).classes("w-full"):
        with ui.tab_panel(run_tab):
            run_views = build_run_tab(state)
        with ui.tab_panel(leaderboard_tab):
            build_leaderboard_tab(state)
        with ui.tab_panel(history_tab):
            build_history_tab(state)

    ui.timer(0.2, lambda: refresh_ui(state, run_views))


def main() -> None:
    ui.run(root=create_ui, reload=False, dark=False)


if __name__ == "__main__":
    main()

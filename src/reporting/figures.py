"""Static README figures, rendered once for a light and once for a dark background."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.lineage.manifest import load_manifests  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"]},
}
DPI = 200
STRATEGY_LABELS = {"drift_triggered": "Retrain when drift is flagged", "static": "Never retrain",
                   "monthly": "Retrain every month"}


def _system_sans() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial"] if n in installed),
                "DejaVu Sans")


FONT = _system_sans()


def _figure(theme: dict, width: float, height: float):
    plt.rcParams["font.family"] = FONT
    return plt.figure(figsize=(width, height), dpi=DPI, facecolor=theme["surface"])


def _frame(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelcolor=theme["secondary"], length=0, labelsize=9)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _titles(fig, theme: dict, title: str, subtitle: str) -> None:
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.18 / h, title, ha="left", va="top", fontsize=13, weight="semibold",
             color=theme["ink"])
    fig.text(0.02, 1 - 0.47 / h, subtitle, ha="left", va="top", fontsize=9.5,
             color=theme["secondary"])


def _legend(fig, theme: dict, handles, labels, y: float) -> None:
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.015, y), ncol=len(labels),
               frameon=False, fontsize=9, labelcolor=theme["secondary"], handlelength=1.6,
               columnspacing=1.6)


def _month_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def _line(ax, x, y, color, **kwargs):
    return ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", marker="o",
                   markersize=4.5, markeredgewidth=0, **kwargs)


def drift_timeline(monthly: pd.DataFrame, threshold: float, path: Path, mode: str) -> None:
    theme = THEMES[mode]
    frame = monthly.assign(when=pd.to_datetime(monthly["month"]))
    static = frame[frame["strategy"] == "static"]

    fig = _figure(theme, 8, 7.2)
    top = 1 - 1.55 / 7.2
    ax_drift = fig.add_axes([0.09, 0.55, 0.88, top - 0.55])
    ax_bias = fig.add_axes([0.09, 0.09, 0.88, 0.36])
    for ax in (ax_drift, ax_bias):
        _frame(ax, theme)
        _month_axis(ax)

    signals = [
        ("seasonal_sales_drift", "Sales vs same month in training data", theme["series"][0]),
        ("seasonal_prediction_drift", "Forecasts vs same month", theme["series"][1]),
        ("naive_sales_drift", "Sales vs whole training window (naive)", theme["series"][2]),
    ]
    for column, _, color in signals:
        _line(ax_drift, static["when"], static[column], color)
    ax_drift.axhline(threshold, color=theme["muted"], linewidth=1)
    ax_drift.annotate(f"drift threshold {threshold}", xy=(1, threshold),
                      xycoords=("axes fraction", "data"), xytext=(-4, 3), ha="right", va="bottom",
                      textcoords="offset points", fontsize=8.5, color=theme["secondary"])
    ax_drift.set_ylim(0, None)
    ax_drift.set_title(
        "Drift scores for the model that is never retrained (normalised Wasserstein)",
        loc="left", fontsize=10, color=theme["ink"], pad=6,
    )
    _legend(fig, theme, [Line2D([], [], color=c, linewidth=2) for _, _, c in signals],
            [label for _, label, _ in signals], y=1 - 0.8 / 7.2)

    # Drift-triggered is drawn last: in 2016 it retrained every month, so it sits exactly on
    # top of the monthly strategy and would otherwise be hidden.
    strategies = ["static", "monthly", "drift_triggered"]
    colors = {"static": theme["series"][1], "monthly": theme["series"][2],
              "drift_triggered": theme["series"][0]}
    for strategy in strategies:
        part = frame[frame["strategy"] == strategy]
        _line(ax_bias, part["when"], part["bias_pct"], colors[strategy])
        last = part.iloc[-1]
        below = strategy == "drift_triggered"  # keeps its label clear of the monthly line
        ax_bias.annotate(STRATEGY_LABELS[strategy], xy=(last["when"], last["bias_pct"]),
                         xytext=(-4, -9 if below else 7), va="top" if below else "baseline",
                         textcoords="offset points", ha="right", fontsize=8.5,
                         color=theme["secondary"])
    ax_bias.axhline(0, color=theme["axis"], linewidth=1)
    retrains = frame[(frame["strategy"] == "drift_triggered") & frame["retrained"].astype(bool)]
    ymin = min(frame[frame["strategy"].isin(strategies)]["bias_pct"].min() * 1.15, -1)
    ax_bias.set_ylim(ymin, max(3, frame["bias_pct"].max() * 1.3))
    ax_bias.scatter(retrains["when"] + pd.Timedelta(days=27), [ymin * 0.96] * len(retrains),
                    marker="^", s=28, color=theme["series"][0], zorder=3, linewidths=0)
    ax_bias.annotate("▲ drift-triggered retrain", xy=(0, 0), xycoords="axes fraction",
                     xytext=(4, 14), textcoords="offset points", ha="left", fontsize=8.5,
                     color=theme["secondary"])
    ax_bias.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
    ax_bias.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax_bias.set_title("Forecast bias each month (total forecast vs total actual sales)",
                      loc="left", fontsize=10, color=theme["ink"], pad=6)

    _titles(fig, theme, "Catching a forecast going stale",
            "500 store-item series, forecast one month at a time through 2016 and 2017.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def drift_is_not_decay(monthly: pd.DataFrame, path: Path, mode: str) -> None:
    theme = THEMES[mode]
    frame = monthly.assign(when=pd.to_datetime(monthly["month"]))
    panels = [("static", "Calendar-only model"),
              ("static_with_recent_sales", "Model with recent-sales features")]
    fig = _figure(theme, 8, 4.4)
    top = 1 - 1.45 / 4.4
    axes = [fig.add_axes([left, 0.14, 0.4, top - 0.14]) for left in (0.08, 0.57)]
    ymax = max(frame[c].max() for c in ("seasonal_sales_drift", "seasonal_prediction_drift")) * 1.1
    for ax, (strategy, label) in zip(axes, panels, strict=True):
        part = frame[frame["strategy"] == strategy]
        _frame(ax, theme)
        _month_axis(ax)
        _line(ax, part["when"], part["seasonal_sales_drift"], theme["series"][0])
        _line(ax, part["when"], part["seasonal_prediction_drift"], theme["series"][1])
        ax.set_ylim(0, ymax)
        bias = part["bias_pct"].mean()
        ax.set_title(f"{label}\naverage bias {bias:+.1f}%", loc="left", fontsize=10,
                     color=theme["ink"], pad=6)
    _legend(fig, theme, [Line2D([], [], color=theme["series"][i], linewidth=2) for i in (0, 1)],
            ["Sales drift", "Forecast drift"], y=1 - 0.78 / 4.4)
    _titles(fig, theme, "Drift isn't the same as decay",
            "Both models see the same growing sales. "
            "Only the one whose forecasts don't follow is broken.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def lineage_timeline(lineage_dir: Path, path: Path, mode: str,
                     strategy: str = "drift_triggered") -> None:
    theme = THEMES[mode]
    manifests = load_manifests(lineage_dir, strategy)
    fig = _figure(theme, 8, 1.6 + 0.27 * len(manifests))
    height = fig.get_figheight()
    ax = fig.add_axes([0.1, 0.55 / height, 0.87, 1 - (1.45 + 0.55) / height])
    _frame(ax, theme)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    for row, m in enumerate(manifests):
        data, served = m["training_data"], m["served"]
        t0, t1 = pd.Timestamp(data["date_from"]), pd.Timestamp(data["date_to"])
        s0, s1 = pd.Timestamp(served["from"]), pd.Timestamp(served["to"])
        ax.barh(row, (t1 - t0).days, left=t0, height=0.5, color=theme["axis"], linewidth=0)
        ax.barh(row, (s1 - s0).days + 1, left=s0, height=0.5, color=theme["series"][0], linewidth=0)
    ax.set_yticks(range(len(manifests)), labels=[m["model_version"] for m in manifests])
    ax.set_ylim(len(manifests) - 0.5, -0.7)
    ax.tick_params(axis="y", labelsize=8.5, labelcolor=theme["ink"])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _legend(fig, theme, [Line2D([], [], color=theme["axis"], linewidth=6),
                         Line2D([], [], color=theme["series"][0], linewidth=6)],
            ["Training data", "Months in production"], y=1 - 0.78 / height)
    _titles(fig, theme, "Every model version, and the data behind it",
            "Retrain-when-drift-is-flagged strategy. Each row is one lineage manifest.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def render_all(results: dict, monthly: pd.DataFrame, lineage_dir: Path,
               figures_dir: Path) -> list[Path]:
    threshold = results["config"]["drift_threshold"]
    written = []
    for mode in THEMES:
        jobs = {
            "drift_timeline": lambda p, m: drift_timeline(monthly, threshold, p, m),
            "drift_is_not_decay": lambda p, m: drift_is_not_decay(monthly, p, m),
            "lineage_timeline": lambda p, m: lineage_timeline(lineage_dir, p, m),
        }
        for name, draw in jobs.items():
            path = figures_dir / f"{name}_{mode}.png"
            draw(path, mode)
            written.append(path)
    return written

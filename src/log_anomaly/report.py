"""Descriptive Phase 2 EDA charts and the Phase 5 annotated incident timeline."""

from pathlib import Path


def write_eda_charts(windows: list[dict], output: Path, timezone: str) -> None:
    # Lazy imports keep the Phase 1 command independent of plotting dependencies.
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from datetime import datetime

    services = sorted({w["service"] for w in windows})
    specs = {
        "volume": ("Log volume", "Records / 5 min", [
            ("business", "record_count", "Business"), ("health", "record_count", "Health")]),
        "status": ("Business HTTP status", "Fraction of valid statuses", [
            ("business", f"http_{n}xx_rate", f"{n}xx") for n in range(1, 6)]),
        "latency": ("Business response latency", "Milliseconds", [
            ("business", "median_latency_ms", "Median"), ("business", "p95_latency_ms", "p95")]),
        "levels": ("Business log levels", "Fraction of records", [
            ("level_rates", level, level) for level in ("INFO", "WARN", "ERROR", "UNKNOWN")]),
    }
    output.mkdir()
    for name, (title, ylabel, series) in specs.items():
        figure = Figure(figsize=(12, 2.5 * len(services)), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(len(services), 1, squeeze=False, sharex=True)
        for index, service in enumerate(services):
            axis = axes[index, 0]
            selected = [w for w in windows if w["service"] == service
                        and w["scope"] == "service" and w["window_minutes"] == 5]
            times = [datetime.fromisoformat(w["window_start"]) for w in selected]
            for population, metric, label in series:
                values = [(w["business"][population][metric] if population == "level_rates"
                           else w[population][metric]) for w in selected]
                axis.plot(times, [float("nan") if v is None else v for v in values],
                          label=label, linewidth=0.8)
            axis.set_title(service, loc="left")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
            if name in {"status", "levels"}:
                axis.set_ylim(-0.03, 1.03)
            if index == 0:
                axis.legend(loc="upper right", ncols=len(series), fontsize="small")
        locator = mdates.AutoDateLocator()
        axes[-1, 0].xaxis.set_major_locator(locator)
        axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[-1, 0].set_xlabel(f"Source local time (timezone: {timezone}); missing metrics shown as gaps")
        figure.suptitle(title)
        figure.savefig(output / f"{name}.png", dpi=120)
        figure.clear()


def write_incident_chart(windows: list[dict], incidents: list[dict], path: Path, timezone: str) -> None:
    """Shade grouped incident spans over five-minute business volume; colour encodes severity only."""
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from datetime import datetime

    colours = {"review": "#d4a017", "warning": "#e66100", "high": "#c0182b"}
    services = sorted({w["service"] for w in windows})
    figure = Figure(figsize=(12, 2.5 * len(services)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(services), 1, squeeze=False, sharex=True)
    for index, service in enumerate(services):
        axis = axes[index, 0]
        selected = [w for w in windows if w["service"] == service
                    and w["scope"] == "service" and w["window_minutes"] == 5]
        axis.plot([datetime.fromisoformat(w["window_start"]) for w in selected],
                  [w["business"]["record_count"] for w in selected], color="#555555", linewidth=0.6)
        # Higher severities are drawn last so they remain visible over overlapping spans.
        for incident in sorted(incidents, key=lambda i: list(colours).index(i["severity"])):
            if service in incident["services"]:
                axis.axvspan(datetime.fromisoformat(incident["start_time"]),
                             datetime.fromisoformat(incident["end_time"]),
                             color=colours[incident["severity"]], alpha=0.35, linewidth=0)
        axis.set_title(service, loc="left")
        axis.set_ylabel("Business records / 5 min")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(handles=[Patch(color=c, alpha=0.35, label=s) for s, c in colours.items()],
                      loc="upper right", ncols=3, fontsize="small", title="Incident severity")
    locator = mdates.AutoDateLocator()
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1, 0].set_xlabel(f"Source local time (timezone: {timezone}); shaded spans are grouped incidents")
    figure.suptitle("Grouped incidents (all scopes) over business volume")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    figure.clear()

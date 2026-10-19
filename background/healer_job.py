"""The nightly healer report (ROADMAP D4; architecture 6.1).

Read-only, always. The architecture's nightly pass merges near-duplicates and prunes orphans;
here it only reports. Gate 8 still shows false merges against gold (WP10), so an unattended
merge would make identity worse, and every change goes through `main_admin.py heal --apply`
after a person has read the report.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAST_KEY = "healer_report_last"


def run_heal_report(settings: Any, store: Any, healer_factory: Any | None = None) -> dict[str, Any]:
    """Run the healer as a dry run, write the full report to the report directory, and return
    its counts."""
    if healer_factory is None:
        from ingestion.healer import GraphHealer

        healer_factory = GraphHealer
    healer = healer_factory(settings, store=store)
    try:
        report = healer.heal(dry_run=True)
    finally:
        healer.close()
    now = datetime.now(timezone.utc)
    directory = Path(settings.report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"healer-{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    full = asdict(report)
    path.write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
    summary = {key: value for key, value in full.items() if key != "contradictions"}
    summary["contradiction_samples"] = len(full.get("contradictions") or [])
    summary["report"] = str(path)
    store.set_system_metadata(LAST_KEY, json.dumps({"at": now.isoformat(), "report": str(path)}))
    return summary


def last_report(store: Any) -> dict[str, Any] | None:
    raw = store.get_system_metadata(LAST_KEY)
    try:
        value = json.loads(raw) if raw else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None

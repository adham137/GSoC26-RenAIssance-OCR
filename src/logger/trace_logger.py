"""
src/logger/trace_logger.py
===========================
Execution Logger & Trace Module (System Design §2.6).

Because agentic loops act as "black boxes," every thought, action,
adapter toggle, and intermediate transcription is recorded here.
Logs are written to:
    * A structured JSON file per page run (for programmatic analysis).
    * The Python ``logging`` module (for console / file output).
    * An in-memory list (returned to the Streamlit UI for rendering).

Log Format
----------
Each :class:`~src.models.AgenticTrace` entry is serialised to JSON with
ISO-8601 timestamps.  The Streamlit UI reads ``agentic_trace`` from the
:class:`~src.models.OCRResult` object directly; this logger also writes
a parallel file for offline analysis.

File naming: ``logs/<page_id>_<timestamp>.jsonl``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.models import AgenticTrace

logger = logging.getLogger(__name__)


class TraceLogger:
    """
    Records and persists agentic trace steps.

    Parameters
    ----------
    logs_dir : str | Path
        Directory where JSONL trace files are written.
    page_id : str
        Identifier for the current page run; used to name the log file.
    echo_to_console : bool
        If ``True``, each trace entry is also emitted via ``logging.debug``.

    Examples
    --------
    >>> tl = TraceLogger(logs_dir="logs/", page_id="manuscript_page_001")
    >>> tl.record(AgenticTrace(iteration=1, step_type="REFLECT", ...))
    """

    def __init__(
        self,
        logs_dir        : str | Path = "logs/",
        page_id         : str        = "unknown_page",
        echo_to_console : bool       = True,
    ) -> None:
        self.logs_dir        = Path(logs_dir)
        self.page_id         = page_id
        self.echo_to_console = echo_to_console
        self._entries        : list[dict] = []

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._log_file = self.logs_dir / f"{page_id}_{ts}.jsonl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, trace: AgenticTrace) -> None:
        """
        Persist a single :class:`~src.models.AgenticTrace` entry.

        Appends to the in-memory list and writes one JSON line to disk.

        Parameters
        ----------
        trace : AgenticTrace
            The trace entry to record.
        """
        entry = {
            "timestamp"      : datetime.now(tz=timezone.utc).isoformat(),
            "iteration"      : trace.iteration,
            "step_type"      : trace.step_type,
            "adapter_state"  : trace.adapter_state,
            "action"         : trace.action,
            "thought_length" : len(trace.thought),
            "observation"    : trace.observation[:500],  # truncate for log file
            "memory_depth"   : len(trace.memory_snapshot),
        }
        self._entries.append(entry)

        with self._log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self.echo_to_console:
            logger.debug(
                "[iter=%d | %s | adapter=%s] %s → %s chars output",
                trace.iteration,
                trace.step_type,
                trace.adapter_state,
                trace.action,
                len(trace.observation),
            )

    def get_all_entries(self) -> list[dict]:
        """
        Return the complete in-memory trace as a list of dicts.

        Returns
        -------
        list[dict]
            All recorded trace entries in order.
        """
        return list(self._entries)

    def summary(self) -> dict:
        """
        Return a high-level summary of the trace.

        Returns
        -------
        dict
            Keys: ``total_steps``, ``iterations``, ``adapter_toggles``,
            ``log_file``.
        """
        toggles = sum(
            1 for i in range(1, len(self._entries))
            if self._entries[i]["adapter_state"] != self._entries[i - 1]["adapter_state"]
        )
        return {
            "total_steps"    : len(self._entries),
            "iterations"     : max((e["iteration"] for e in self._entries), default=0),
            "adapter_toggles": toggles,
            "log_file"       : str(self._log_file),
        }

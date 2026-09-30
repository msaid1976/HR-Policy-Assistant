"""Central logging with one folder per day and app session.

Log files are stored as ``logs/YYYY-MM-DD/<session-id>/app.log``. Every
execution receives a session identifier. The active identifier is held in a
``ContextVar`` so concurrent Streamlit sessions never write to each other's
log files.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
import logging
from pathlib import Path
import re
from uuid import uuid4


LOGS_DIR = Path("logs")
_SESSION_ID: ContextVar[str | None] = ContextVar("log_session_id", default=None)
_SAFE_ID = re.compile(r"[^a-zA-Z0-9_-]+")


def _safe_session_id(session_id: str) -> str:
    """Make an identifier safe to use as a single folder name."""
    cleaned = _SAFE_ID.sub("-", session_id).strip("-_")
    return cleaned or f"session_{uuid4().hex[:12]}"


def _active_session_id() -> str | None:
    """Return the active ID, or ``None`` before an entry point starts one."""
    return _SESSION_ID.get()


def start_session_logging(session_id: str | None = None) -> str:
    """Set the log destination for the current Streamlit session.

    Pass an authenticated user ID here in the future if authentication is
    added. Until then, a generated session ID keeps separate browser sessions
    isolated without recording personally identifiable information.
    """
    active_id = _safe_session_id(session_id or f"session_{uuid4().hex[:12]}")
    _SESSION_ID.set(active_id)
    return active_id


class DailySessionFileHandler(logging.Handler):
    """Write each record to the folder for its current date and session."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            timestamp = datetime.fromtimestamp(record.created)
            session_id = _active_session_id()
            if session_id is None:
                return
            log_path = LOGS_DIR / timestamp.strftime("%Y-%m-%d") / session_id / "app.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            message = self.format(record)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{message}\n")
        except Exception:
            self.handleError(record)


def _configure_root_logger() -> None:
    root_logger = logging.getLogger()
    if any(isinstance(handler, DailySessionFileHandler) for handler in root_logger.handlers):
        return

    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = DailySessionFileHandler()
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


_configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Return a logger that routes records to the active run/session folder."""
    return logging.getLogger(name)

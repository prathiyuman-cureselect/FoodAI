"""
NeuroVitals — Structured Audit Logging (FHIR-aligned)
=====================================================
JSON-structured audit events with rotating file handler, aligned with
FHIR AuditEvent resource fields for regulatory compatibility.
"""

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Logger Configuration
# ---------------------------------------------------------------------------

_LOG_DIR = os.environ.get("NEUROVITALS_LOG_DIR",
                          os.path.join(os.path.dirname(__file__), "..", "_logs"))


def configure_audit_logger(name: str = "neurovitals.audit",
                           log_dir: Optional[str] = None,
                           max_bytes: int = 10 * 1024 * 1024,
                           backup_count: int = 5) -> logging.Logger:
    """Set up a JSON audit logger with rotating file + stream output.

    Parameters
    ----------
    name : str           Logger name.
    log_dir : str | None Directory for log files. Defaults to ``_logs/``.
    max_bytes : int      Max log-file size before rotation (default 10 MB).
    backup_count : int   Number of rotated files to keep.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")

    # Stream handler (console)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Rotating file handler
    log_dir = log_dir or _LOG_DIR
    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(log_dir, "audit.jsonl"),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass  # environments without write access — stream-only

    return logger


# ---------------------------------------------------------------------------
# FHIR-aligned Audit Event Builder
# ---------------------------------------------------------------------------

def _build_event(event_type: str,
                 subtype: str,
                 data: Dict[str, Any],
                 agent: str = "system",
                 outcome: str = "success") -> str:
    """Build a JSON audit event string aligned with FHIR AuditEvent."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "subtype": subtype,
        "agent": agent,
        "source": "NeuroVitals-rPPG-SDK",
        "outcome": outcome,
        "entity": data,
    }
    return json.dumps(payload, default=str)


def audit_event(logger: logging.Logger, event_type: str,
                data: Dict[str, Any], **kwargs) -> None:
    """Log a generic audit event (backwards-compatible)."""
    logger.info(_build_event(event_type, kwargs.get("subtype", "general"),
                             data,
                             agent=kwargs.get("agent", "system"),
                             outcome=kwargs.get("outcome", "success")))


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def audit_analysis(logger: logging.Logger, data: Dict[str, Any],
                   agent: str = "system") -> None:
    """Log a completed analysis event."""
    logger.info(_build_event("analysis", "rppg-pipeline", data, agent=agent))


def audit_identity(logger: logging.Logger, data: Dict[str, Any],
                   agent: str = "system",
                   outcome: str = "success") -> None:
    """Log an identity verification event."""
    logger.info(_build_event("identity", "face-verify", data,
                             agent=agent, outcome=outcome))


def audit_consent(logger: logging.Logger, data: Dict[str, Any],
                  agent: str = "system") -> None:
    """Log a consent token event (grant / revoke)."""
    logger.info(_build_event("consent", "token-lifecycle", data, agent=agent))


def audit_governance(logger: logging.Logger, data: Dict[str, Any],
                     agent: str = "system") -> None:
    """Log a governance / drift / bias monitoring event."""
    logger.info(_build_event("governance", "monitoring", data, agent=agent))

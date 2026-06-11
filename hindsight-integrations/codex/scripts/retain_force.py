#!/usr/bin/env python3
"""Auto-retain hook for Stop event.

Fires after each agent turn. Reads the Codex session transcript and stores
the conversation into Hindsight memory for future recall.

Flow:
  1. Read hook input from stdin (session_id, transcript_path, cwd)
  2. Read conversation transcript from transcript_path
  3. Apply chunked retention logic (retainEveryNTurns + overlap window)
  4. Resolve API URL (external, existing local, or auto-start daemon)
  5. Derive bank ID and ensure mission
  6. Format transcript (strip memory tags, filter roles)
  7. POST to Hindsight retain API

Exit codes:
  0 — always (graceful degradation on any error)
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.content import (
    prepare_retention_transcript,
    read_transcript,
    slice_last_turns_by_user_boundary,
)
from lib.daemon import get_api_url
from lib.state import increment_turn_count


def find_transcript(session_id: str):
    sessions_root = Path.home() / ".codex" / "sessions"

    candidates = list(sessions_root.rglob(f"*{session_id}*.jsonl"))
    if not candidates:
        return None, None

    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as f:
                first = json.loads(f.readline())
            payload = first.get("payload", {})
            if first.get("type") == "session_meta" and payload.get("id") == session_id:
                return path, payload.get("cwd", os.getcwd())
        except Exception:
            continue

    return None, None

def main():
    # Create hook input
    try:
        sessions_root = Path.home() / ".codex" / "sessions"
        retain_script = Path.home() / ".hindsight" / "codex" / "scripts" / "retain.py"

        latest = max(sessions_root.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime)

        with latest.open("r", encoding="utf-8") as f:
            first = json.loads(f.readline())

        session_id = first["payload"]["id"]
        cwd = first["payload"].get("cwd", os.getcwd())

        hook_input = {
            "session_id": session_id,
            "transcript_path": str(latest),
            "cwd": cwd
        }

        session_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CODEX_THREAD_ID")
        if not session_id:
            print("[Hindsight] CODEX_THREAD_ID not found; pass session_id manually", file=sys.stderr)
            sys.exit(1)

        transcript_path, cwd = find_transcript(session_id)
        if not transcript_path:
            print(f"[Hindsight] Transcript not found for session_id={session_id}", file=sys.stderr)
            sys.exit(1)

        hook_input = {
            "session_id": session_id,
            "transcript_path": str(transcript_path),
            "cwd": cwd
        }
        print(f"session_id: {session_id}, transcript_path: {str(latest)}, cwd: {cwd}")
    except (json.JSONDecodeError, EOFError):
        print("[Hindsight] Failed to read hook input", file=sys.stderr)
        return

    config = load_config(cwd=hook_input.get("cwd"))
    debug_log(config, f"Stop hook input keys: {list(hook_input.keys())}")

    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path", "")

    # Read full transcript
    include_tool_calls = config.get("retainToolCalls", True)
    all_messages = read_transcript(transcript_path, include_tool_calls=include_tool_calls)
    if not all_messages:
        debug_log(config, "No messages in transcript, skipping retain")
        return

    debug_log(config, f"Read {len(all_messages)} messages from transcript")

    # Retention mode: full session (default) or chunked (legacy)
    retain_mode = config.get("retainMode", "full-session")
    retain_every_n = max(1, config.get("retainEveryNTurns", 1))
    retain_full_window = False
    messages_to_retain = all_messages

    if retain_mode == "chunked" and retain_every_n > 1:
        overlap_turns = config.get("retainOverlapTurns", 0)
        window_turns = retain_every_n + overlap_turns
        messages_to_retain = slice_last_turns_by_user_boundary(all_messages, window_turns)
        retain_full_window = True
        debug_log(
            config,
            f"Chunked retain firing (window: {window_turns} turns, {len(messages_to_retain)} messages)",
        )
    else:
        retain_full_window = True
        debug_log(config, f"Full session retain: {len(all_messages)} messages")

    # Format transcript
    retain_roles = config.get("retainRoles", ["user", "assistant"])
    transcript, message_count = prepare_retention_transcript(
        messages_to_retain, retain_roles, retain_full_window, include_tool_calls=include_tool_calls
    )

    if not transcript:
        debug_log(config, "Empty transcript after formatting, skipping retain")
        return

    # Resolve API URL
    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=True)
    except RuntimeError as e:
        print(f"[Hindsight] {e}", file=sys.stderr)
        return

    api_token = config.get("hindsightApiToken")
    try:
        client = HindsightClient(api_url, api_token)
    except ValueError as e:
        print(f"[Hindsight] Invalid API URL: {e}", file=sys.stderr)
        return

    # Derive bank ID and ensure mission
    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    # Document ID: use session_id so the same session always upserts.
    # In chunked mode, append timestamp to create distinct documents per chunk.
    if retain_mode == "chunked" and retain_every_n > 1:
        document_id = f"{session_id}-{int(time.time() * 1000)}"
    else:
        document_id = session_id

    # Resolve template variables in tags and metadata
    template_vars = {
        "session_id": session_id,
        "bank_id": bank_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def _resolve_template(value: str) -> str:
        for k, v in template_vars.items():
            value = value.replace(f"{{{k}}}", v)
        return value

    raw_tags = config.get("retainTags", [])
    tags = [_resolve_template(t) for t in raw_tags] if raw_tags else None

    metadata = {
        "retained_at": template_vars["timestamp"],
        "message_count": str(message_count),
        "session_id": session_id,
    }
    for k, v in config.get("retainMetadata", {}).items():
        metadata[k] = _resolve_template(str(v))

    debug_log(
        config, f"Retaining to bank '{bank_id}', doc '{document_id}', {message_count} messages, {len(transcript)} chars"
    )
    if tags:
        debug_log(config, f"Tags: {tags}")

    # POST to Hindsight retain API
    try:
        response = client.retain(
            bank_id=bank_id,
            content=transcript,
            document_id=document_id,
            context=config.get("retainContext", "codex"),
            metadata=metadata,
            tags=tags,
            timeout=15,
        )
        debug_log(config, f"Retain response: {json.dumps(response)[:200]}")
    except Exception as e:
        print(f"[Hindsight] Retain failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] Unexpected error in retain: {e}", file=sys.stderr)
        try:
            from lib.config import load_config

            sys.exit(2 if load_config().get("debug") else 0)
        except Exception:
            sys.exit(0)

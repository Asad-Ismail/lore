"""
Claude Code PostToolUse hook: capture /wiki query interactions as RL trajectories.

This script is called by Claude Code after every Bash tool use.
It receives tool input/output as JSON on stdin and checks if the
command was a wiki query that should be captured.

Registration in ~/.claude/settings.json:
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Bash",
      "hooks": [{"type": "command",
        "command": "python3 /home/ec2-user/SageMaker/personal-knowledge/hooks/capture_trajectory.py"}]
    }]
  }
}
"""

from __future__ import annotations

import json
import sys
import os
import re

# Add project to path
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def main():
    # Read hook input from stdin
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        data = json.loads(raw)
    except Exception:
        return

    # Check if this was a wiki query command
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    tool_output = data.get("tool_response", {})

    if tool_name != "Bash":
        return

    command = tool_input.get("command", "")

    # Detect lore query patterns
    is_wiki_query = any(pattern in command for pattern in [
        "lore-query", "lore query",
        "answer_question", "lore.query",
    ])
    if not is_wiki_query:
        return

    # Extract question from the query command
    question = _extract_question(command)
    if not question:
        return

    # Extract response from tool output
    output_text = ""
    if isinstance(tool_output, dict):
        output_text = tool_output.get("output", "") or tool_output.get("stdout", "")
    elif isinstance(tool_output, str):
        output_text = tool_output

    if not output_text.strip():
        return

    # Capture minimal trajectory (no retrieved context available from hook)
    try:
        from lore.evolve.trajectory import Trajectory, save_trajectory
        from lore.evolve.reward import compute_instant_rewards

        traj = Trajectory(
            question=question,
            retrieved_paths=[],
            context="",  # Context not available from hook output
            response=output_text,
            citations=_extract_citations(output_text),
            citation_validation={},
            metadata={"source": "hook", "command": command[:200]},
        )

        instant = compute_instant_rewards(traj)
        traj.reward = instant["combined_partial"]
        traj.metadata.update({
            "grounding": instant["grounding"],
            "fluency": instant["fluency"],
        })

        save_trajectory(traj)
        print(f"[hook] Trajectory captured: {traj.id[:8]} | reward={traj.reward:.3f}",
              file=sys.stderr)
    except Exception as e:
        print(f"[hook] Trajectory capture error: {e}", file=sys.stderr)


def _extract_question(command: str) -> str | None:
    """Extract the question from a lore query command string."""
    # lore-query "question here"
    m = re.search(r'lore[_-]query\s+"([^"]+)"', command)
    if m:
        return m.group(1)
    # lore query question here (unquoted)
    m = re.search(r'lore[_-]query\s+(.+)', command)
    if m:
        return m.group(1).strip()
    return None


def _extract_citations(text: str) -> list[str]:
    """Extract [[WikiLink]] citations from response text."""
    return re.findall(r"\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]", text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tests for PROCESS_VIOLATION clustering in the candidate aggregator.

Builds three synthetic events per violation kind (nine violations total,
three kinds) and asserts that the aggregator collapses each kind into the
expected number of clustered candidates.

The tests call the clustering helpers directly on in-memory event rows —
no sqlite DB access is performed. Note that `CandidateAggregator.__init__`
still reads `~/.claude-coach/config.json` if it exists (via `_load_config`);
the tests tolerate either an existing config file or an absent one, so
running them on a developer machine with a live coach install is safe.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Make scripts/ importable so test can be run from any cwd.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from aggregate import CandidateAggregator  # noqa: E402


def _make_event(
    event_id: str, violations, repo_id="repo-a", timestamp="2026-04-18T09:00:00Z"
):
    """Build a minimal event row matching the events table shape."""
    return {
        "id": event_id,
        "timestamp": timestamp,
        "event_type": "tool",
        "signal_type": "PROCESS_VIOLATION",
        "repo_id": repo_id,
        "content": json.dumps(
            {
                "signal_type": "PROCESS_VIOLATION",
                "violations": violations,
                "confidence": 0.8,
            }
        ),
        "context": "{}",
        "processed": 0,
    }


class ProcessViolationClusteringTests(unittest.TestCase):
    def setUp(self):
        # Disable transcript analyzer so constructing the aggregator is cheap
        # and doesn't touch ~/.claude.
        self.aggregator = CandidateAggregator(analyze_transcript=False)

    def test_unauthorized_squash_clusters_by_base_command(self):
        """Three squash violations across two base commands -> two candidates."""
        events = [
            _make_event(
                "ev-sq-1",
                [
                    {
                        "kind": "unauthorized_squash",
                        "pattern": r"\bgh\s+pr\s+merge\b[^\n]*--squash\b",
                        "command": "gh pr merge 42 --squash --delete-branch",
                    }
                ],
            ),
            _make_event(
                "ev-sq-2",
                [
                    {
                        "kind": "unauthorized_squash",
                        "pattern": r"\bgh\s+pr\s+merge\b[^\n]*--squash\b",
                        "command": "gh pr merge --squash --auto",
                    }
                ],
            ),
            _make_event(
                "ev-sq-3",
                [
                    {
                        "kind": "unauthorized_squash",
                        "pattern": r"\bgit\s+merge\s+--squash\b",
                        "command": "git merge --squash feature-branch",
                    }
                ],
            ),
        ]

        # Force repo-policy lookup to a known value so the test is deterministic
        # regardless of the cwd gh happens to see.
        with patch.object(
            CandidateAggregator, "_repo_allows_squash", staticmethod(lambda: False)
        ):
            candidates = self.aggregator.extract_candidate_from_process_violation(
                events
            )

        self.assertEqual(len(candidates), 2, candidates)
        kinds = {c["violation_kind"] for c in candidates}
        self.assertEqual(kinds, {"unauthorized_squash"})

        by_base = {c["base_command"]: c for c in candidates}
        self.assertIn("gh pr merge", by_base)
        self.assertIn("git merge --squash", by_base)

        gh_cand = by_base["gh pr merge"]
        self.assertEqual(gh_cand["occurrence_count"], 2)
        self.assertEqual(gh_cand["candidate_type"], "antipattern")
        self.assertEqual(gh_cand["repo_allow_squash_merge"], False)
        # Fingerprint must be stable & unique per base command.
        self.assertNotEqual(
            gh_cand["fingerprint"],
            by_base["git merge --squash"]["fingerprint"],
        )

    def test_cache_path_edit_clusters_by_path_prefix(self):
        """Three cache-path edits across two prefixes -> two candidates."""
        events = [
            _make_event(
                "ev-cp-1",
                [
                    {
                        "kind": "cache_path_edit",
                        "pattern": r"[/~]\.claude/skills/",
                        "tool": "Edit",
                        "file_path": "/home/x/.claude/skills/coach/SKILL.md",
                    }
                ],
            ),
            _make_event(
                "ev-cp-2",
                [
                    {
                        "kind": "cache_path_edit",
                        "pattern": r"[/~]\.claude/skills/",
                        "tool": "Write",
                        "file_path": "~/.claude/skills/coach/scripts/x.py",
                    }
                ],
            ),
            _make_event(
                "ev-cp-3",
                [
                    {
                        "kind": "cache_path_edit",
                        "pattern": r"/\.bare/",
                        "tool": "Edit",
                        "file_path": "/home/cybot/projects/foo/.bare/HEAD",
                    }
                ],
            ),
        ]

        candidates = self.aggregator.extract_candidate_from_process_violation(events)

        self.assertEqual(len(candidates), 2, candidates)
        by_prefix = {c["path_prefix"]: c for c in candidates}
        self.assertIn("~/.claude/skills/", by_prefix)
        self.assertIn("/.bare/", by_prefix)

        skills = by_prefix["~/.claude/skills/"]
        self.assertEqual(skills["occurrence_count"], 2)
        self.assertEqual(skills["distinct_file_count"], 2)
        self.assertEqual(skills["candidate_type"], "rule")
        self.assertEqual(sorted(skills["tools_used"]), ["Edit", "Write"])

        # Fingerprints differ across clusters.
        self.assertNotEqual(skills["fingerprint"], by_prefix["/.bare/"]["fingerprint"])

    def test_premature_success_claim_clusters_by_session(self):
        """Three claim violations across two (repo, day) sessions -> two candidates."""
        events = [
            _make_event(
                "ev-ps-1",
                [
                    {
                        "kind": "premature_success_claim",
                        "pattern": r"\bverified\b",
                        "snippet": "I've verified the changes — all tests pass.",
                    }
                ],
                repo_id="repo-a",
                timestamp="2026-04-18T10:00:00Z",
            ),
            _make_event(
                "ev-ps-2",
                [
                    {
                        "kind": "premature_success_claim",
                        "pattern": r"\bshould\s+work\s+now\b",
                        "snippet": "It should work now. Try again.",
                    }
                ],
                repo_id="repo-a",
                timestamp="2026-04-18T14:30:00Z",
            ),
            _make_event(
                "ev-ps-3",
                [
                    {
                        "kind": "premature_success_claim",
                        "pattern": r"\ball\s+green\b",
                        # Intentionally long to exercise the redaction path.
                        "snippet": "all green " + ("x" * 400),
                    }
                ],
                repo_id="repo-b",
                timestamp="2026-04-18T11:00:00Z",
            ),
        ]

        candidates = self.aggregator.extract_candidate_from_process_violation(events)

        self.assertEqual(len(candidates), 2, candidates)
        by_session = {c["session_key"]: c for c in candidates}
        self.assertIn("repo-a:2026-04-18", by_session)
        self.assertIn("repo-b:2026-04-18", by_session)

        a = by_session["repo-a:2026-04-18"]
        self.assertEqual(a["occurrence_count"], 2)
        self.assertEqual(a["candidate_type"], "checklist")
        # Snippets for the two repo-a events are preserved (both <200 chars).
        self.assertEqual(len(a["snippets"]), 2)

        b = by_session["repo-b:2026-04-18"]
        self.assertEqual(b["occurrence_count"], 1)
        # Long snippet should be truncated with redaction marker.
        self.assertTrue(any("[redacted]" in s for s in b["snippets"]), b["snippets"])

    def test_unknown_kind_falls_through(self):
        """Unknown violation kinds get a generic pass-through candidate."""
        events = [
            _make_event(
                "ev-unk-1",
                [
                    {
                        "kind": "future_kind_xyz",
                        "pattern": r"\bfoo\b",
                        "command": "foo bar",
                    }
                ],
            ),
        ]
        candidates = self.aggregator.extract_candidate_from_process_violation(events)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["violation_kind"], "future_kind_xyz")
        self.assertEqual(candidates[0]["candidate_type"], "rule")

    def test_fingerprint_stable_across_runs(self):
        """Running clustering twice on the same input yields the same fingerprints."""
        events = [
            _make_event(
                "ev-stable",
                [
                    {
                        "kind": "cache_path_edit",
                        "pattern": r"[/~]\.claude/skills/",
                        "tool": "Edit",
                        "file_path": "~/.claude/skills/coach/SKILL.md",
                    }
                ],
            ),
        ]
        first = self.aggregator.extract_candidate_from_process_violation(events)
        second = self.aggregator.extract_candidate_from_process_violation(events)
        self.assertEqual(first[0]["fingerprint"], second[0]["fingerprint"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

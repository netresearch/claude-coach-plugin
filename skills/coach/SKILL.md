---
name: coach
description: "Use when Claude makes repeated mistakes, users correct behavior, tool failures indicate missing knowledge, or managing learning via /coach commands."
license: "(MIT AND CC-BY-SA-4.0). See LICENSE-MIT and LICENSE-CC-BY-SA-4.0"
compatibility: "Requires python3 (for signal detection, aggregation, and analysis scripts)."
metadata:
  author: Netresearch DTT GmbH
  version: "2.5.0"
  repository: https://github.com/netresearch/claude-coach-plugin
allowed-tools: Bash(python3:*) Bash(python:*) Bash(sqlite3:*) Read Write Glob Grep
---

# Coach - Self-Improving Learning System

Coach enables Claude to learn from friction and improve over time. It detects learning opportunities (user corrections, repeated instructions, tool failures, tone escalation), extracts actionable improvement candidates, and proposes changes requiring explicit user approval.

**Core Principle**: No silent writes. All improvements require user approval via `/coach approve`.

## Activation Triggers

Activate when: user corrections ("no", "stop", "don't"), repeated instructions, tool/command failures, tone escalation (ALL CAPS, "!!!"), skill supplements ("also remember..."), deprecated-tool warnings, explicit `/coach` commands, or session end.

## Signal Categories (Priority Order)

1. **COMMAND_FAILURE** (Highest) - Non-zero exit, stderr patterns
2. **USER_CORRECTION** (High) - Explicit correction language
3. **SKILL_SUPPLEMENT** (High) - Additional guidance for a skill
4. **VERSION_ISSUE** (Medium-High) - Deprecated/outdated warnings
5. **REPETITION** (Medium) - Semantically similar instruction repeated
6. **TONE_ESCALATION** (Low) - Frustration indicators

## Candidate Types

| Type | Description | Example |
|------|-------------|---------|
| `rule` | Stable constraint | "Never edit generated files" |
| `checklist` | Workflow step | "Run tests after code change" |
| `snippet` | Repeatable command | "Preflight check script" |
| `skill` | Skill update suggestion | "Add X guidance to Y skill" |
| `antipattern` | Things to never do | "Never assume tool exists" |

## Workflow Summary

1. **Detection** — hooks capture events → `~/.claude-coach/events.sqlite`
2. **Generation** — aggregate signals into proposals (fingerprints dedupe)
3. **Scope** — project vs global per path/language
4. **Review** — `/coach review` approves/rejects/edits
5. **Apply** — approved rules → CLAUDE.md
6. **Retro** — `/coach retro` maps manual work to skill gaps, opens PRs at source repos

## File Locations

```
~/.claude-coach/
├── events.sqlite      # Raw friction events
├── candidates.json    # Pending proposals
└── ledger.sqlite      # Cross-repo fingerprints

~/.claude/ or <repo>/.claude/
├── CLAUDE.md          # Rules destination
├── checklists/        # Workflow checklists
└── snippets/          # Reusable commands
```

## Scripts

Execute from `${CLAUDE_PLUGIN_ROOT}/scripts/`:

| Script | Purpose |
|--------|---------|
| `init_coach.py` | Initialize coach system |
| `detect_signals.py` | Pattern detection for friction |
| `aggregate.py` | Turn signals into candidates |
| `skill_analyzer.py` | Analyze skills and scan for outdated tools |
| `apply.py` | Apply approved proposals |

## Proactive Commands

- `/coach scan` — Check for outdated CLI tools and dependencies
- `/coach retro` — Session retrospective: analyze manual work, create PRs to improve skills at source

For detailed architecture, schemas, and patterns, see `references/` directory.

---
name: coach
description: "Use when Claude makes repeated mistakes, users correct behavior, tool failures indicate missing knowledge, or managing learning via /coach commands."
---

# Coach - Self-Improving Learning System

Coach enables Claude to learn from friction and improve over time. It detects learning opportunities (user corrections, repeated instructions, tool failures, tone escalation), extracts actionable improvement candidates, and proposes changes requiring explicit user approval.

**Core Principle**: No silent writes. All improvements require user approval via `/coach approve`.

## Activation Triggers

Activate this skill when:
- User corrects Claude's behavior ("no", "stop", "don't", "I said", "you didn't")
- Same instruction is repeated within recent turns
- Tool/command failures occur (non-zero exit, stderr patterns)
- Tone escalation detected (ALL CAPS, "!!!", "for the last time")
- User supplements a skill with additional instructions ("the skill doesn't...", "also remember...")
- Deprecated/outdated tool warnings appear in command output
- User explicitly requests `/coach` commands
- Session end triggers batch review of accumulated signals

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

1. **Signal Detection** - Hooks capture friction events → stored in `~/.claude-coach/events.sqlite`
2. **Candidate Generation** - Aggregate signals into proposals with fingerprints for deduplication
3. **Scope Decision** - Determine project vs global scope based on path/language patterns
4. **Proposal Review** - User reviews via `/coach review`, approves/rejects/edits
5. **Application** - Approved rules added to CLAUDE.md (project or global)
6. **Session Retrospective** - `/coach retro` analyzes completed sessions, maps manual work to skill gaps, and creates PRs to improve skills at their source repos

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

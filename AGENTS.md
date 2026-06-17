# AGENTS.md — claude-coach-plugin

Repo index for AI agents. This is a compact map; follow the links for detail.

## What this is

Coach is a **Feature Plugin** for Claude Code: a self-improving learning system
that detects friction signals (user corrections, repeated instructions, tool
failures, tone escalation), extracts improvement candidates, and proposes
rule/skill updates through an explicit approval workflow. No silent writes —
every change requires `/coach approve`.

Distributed both as a Claude Code plugin (hooks + slash commands + skill) and as
a Composer `ai-agent-skill` package.

## Entry points

| Path | What it is |
|------|------------|
| [README.md](README.md) | Human-facing overview, install, command list |
| [.claude-plugin/plugin.json](.claude-plugin/plugin.json) | Plugin manifest (name, version, skills) |
| [composer.json](composer.json) | Composer `ai-agent-skill` package manifest |
| [hooks/hooks.json](hooks/hooks.json) | Hook wiring (UserPromptSubmit, PostToolUse, Stop) |

## Skill

The single skill lives under [skills/coach/](skills/coach/):

- [skills/coach/SKILL.md](skills/coach/SKILL.md) — skill definition: activation
  triggers, signal categories, candidate types, workflow, file locations.

## Slash commands

Command docs in [commands/](commands/) — one Markdown file per `/coach` subcommand:

| Command | Doc |
|---------|-----|
| `/coach status` | [commands/status.md](commands/status.md) |
| `/coach review` | [commands/review.md](commands/review.md) |
| `/coach approve` | [commands/approve.md](commands/approve.md) |
| `/coach reject` | [commands/reject.md](commands/reject.md) |
| `/coach edit` | [commands/edit.md](commands/edit.md) |
| `/coach promote` | [commands/promote.md](commands/promote.md) |
| `/coach scan` | [commands/scan.md](commands/scan.md) |
| `/coach retro` | [commands/retro.md](commands/retro.md) |
| `/coach init` | [commands/init.md](commands/init.md) |

## Scripts

Python implementation in [scripts/](scripts/), invoked via
`${CLAUDE_PLUGIN_ROOT}/scripts/`. Requires `python3` (no third-party deps).

| Script | Purpose |
|--------|---------|
| [scripts/init_coach.py](scripts/init_coach.py) | Initialize the coach system |
| [scripts/detect_signals.py](scripts/detect_signals.py) | Detect friction signals from hooks |
| [scripts/aggregate.py](scripts/aggregate.py) | Turn signals into candidates |
| [scripts/propose.py](scripts/propose.py) | Build proposals from candidates |
| [scripts/apply.py](scripts/apply.py) | Apply approved proposals |
| [scripts/skill_analyzer.py](scripts/skill_analyzer.py) | Analyze skills, scan for outdated tools |
| [scripts/scope_analyzer.py](scripts/scope_analyzer.py) | Project vs global scope heuristics |
| [scripts/root_cause_analyzer.py](scripts/root_cause_analyzer.py) | Cluster failures into root causes |
| [scripts/fingerprint.py](scripts/fingerprint.py) | Dedupe proposals by fingerprint |
| [scripts/ledger.py](scripts/ledger.py) | Cross-repo fingerprint ledger |
| [scripts/hook_healer.py](scripts/hook_healer.py) | Repair stale hook paths |

Tests live in [scripts/tests/](scripts/tests/).

## Launchers & hooks

| Path | What it is |
|------|------------|
| [bin/coach-run](bin/coach-run) | Stable launcher installed to `~/.claude-coach/bin` |
| [bin/setup-hooks](bin/setup-hooks) | Install local git hooks |
| [Build/hooks/pre-push](Build/hooks/pre-push) | pre-push git hook |
| [Build/Scripts/check-plugin-version.sh](Build/Scripts/check-plugin-version.sh) | Verify plugin.json version bump |

## Reference docs

Deep detail in [references/](references/):

- [references/architecture.md](references/architecture.md) — layers: signal collection, aggregation, application
- [references/schema.md](references/schema.md) — SQLite event/ledger and candidate schemas
- [references/scope_heuristics.md](references/scope_heuristics.md) — project vs global rule placement
- [references/signal_patterns.md](references/signal_patterns.md) — friction-signal pattern catalog

## Candidate templates

Output templates in [assets/templates/](assets/templates/):
[rule.md](assets/templates/rule.md),
[checklist.md](assets/templates/checklist.md),
[snippet.md](assets/templates/snippet.md),
[antipattern.md](assets/templates/antipattern.md).

## Runtime data (not in repo)

Coach writes to the user's home, never into the repo:

- `~/.claude-coach/events.sqlite` — raw friction events
- `~/.claude-coach/candidates.json` — pending proposals
- `~/.claude-coach/ledger.sqlite` — cross-repo fingerprints
- `~/.claude/CLAUDE.md` — global rules destination
- `<repo>/AGENTS.md` — project rules destination

## CI

Workflows in [.github/workflows/](.github/workflows/): skill validation, eval
validation, harness verification, template-drift, security, scorecard, release.

## Licensing

Split: code is [MIT](LICENSE-MIT); content (skill, docs, references) is
[CC-BY-SA-4.0](LICENSE-CC-BY-SA-4.0).

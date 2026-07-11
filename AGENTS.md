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

Python implementation in [skills/coach/scripts/](skills/coach/scripts/), invoked
via `${CLAUDE_PLUGIN_ROOT}/skills/coach/scripts/`. Nested under `skills/coach/`
so a skill-dir-only install (composer `ai-agent-skill`, release download) ships
a self-contained tree. Requires `python3` (no third-party deps).

| Script | Purpose |
|--------|---------|
| [scripts/init_coach.py](skills/coach/scripts/init_coach.py) | Initialize the coach system |
| [scripts/detect_signals.py](skills/coach/scripts/detect_signals.py) | Detect friction signals from hooks |
| [scripts/aggregate.py](skills/coach/scripts/aggregate.py) | Turn signals into candidates |
| [scripts/propose.py](skills/coach/scripts/propose.py) | Build proposals from candidates |
| [scripts/apply.py](skills/coach/scripts/apply.py) | Apply approved proposals |
| [scripts/skill_analyzer.py](skills/coach/scripts/skill_analyzer.py) | Analyze skills, scan for outdated tools |
| [scripts/scope_analyzer.py](skills/coach/scripts/scope_analyzer.py) | Project vs global scope heuristics |
| [scripts/root_cause_analyzer.py](skills/coach/scripts/root_cause_analyzer.py) | Cluster failures into root causes |
| [scripts/fingerprint.py](skills/coach/scripts/fingerprint.py) | Dedupe proposals by fingerprint |
| [scripts/ledger.py](skills/coach/scripts/ledger.py) | Cross-repo fingerprint ledger |
| [scripts/hook_healer.py](skills/coach/scripts/hook_healer.py) | Repair stale hook paths |

Tests live in [skills/coach/scripts/tests/](skills/coach/scripts/tests/).

## Launchers & hooks

| Path | What it is |
|------|------------|
| [bin/coach-run](bin/coach-run) | Stable launcher installed to `~/.claude-coach/bin` |
| [bin/setup-hooks](bin/setup-hooks) | Install local git hooks |
| [Build/hooks/pre-push](Build/hooks/pre-push) | pre-push git hook |
| [Build/Scripts/check-plugin-version.sh](Build/Scripts/check-plugin-version.sh) | Verify plugin.json version bump |

## Reference docs

Deep detail in [skills/coach/references/](skills/coach/references/):

- [architecture.md](skills/coach/references/architecture.md) — layers: signal collection, aggregation, application
- [schema.md](skills/coach/references/schema.md) — SQLite event/ledger and candidate schemas
- [scope_heuristics.md](skills/coach/references/scope_heuristics.md) — project vs global rule placement
- [signal_patterns.md](skills/coach/references/signal_patterns.md) — friction-signal pattern catalog

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

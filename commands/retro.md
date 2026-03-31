---
description: "Session retrospective — analyze completed work, map manual fixes to skill/checkpoint gaps, and create PRs to improve skills at their source repos."
---

# /coach:retro — Session Retrospective

Analyzes the current session to identify what should have been caught by skills and checkpoints but wasn't, then creates PRs to fix the skill ecosystem.

## When to Use

Run at the end of any session where significant manual work was done that felt like it should have been automated or caught by existing tooling.

## Workflow

### Phase 1: Session Analysis

1. Review the conversation history for all manual fixes, corrections, and improvements made
2. Categorize each into: code fix, infrastructure gap, configuration issue, test gap, anti-pattern
3. For each item, identify which skill SHOULD have caught it

### Phase 2: Skill Gap Mapping

4. List all installed skills: `ls ~/.agents/skills/`
5. For each relevant skill, read its `checkpoints.yaml` and check:
   - Does a checkpoint exist for this issue? → If yes, why didn't it trigger? (skill wasn't invoked, checkpoint too weak, wrong pattern)
   - Does no checkpoint exist? → This is a gap to fill
6. Check skill trigger descriptions — would the session's initial prompt have triggered the right skills?
7. Check the agent-harness skill — does it enforce the quality delegation?

### Phase 3: Fix Plan

8. Group fixes by skill repo
9. For each skill repo, plan:
   - New checkpoints to add (mechanical and LLM)
   - Existing checkpoints to strengthen (patterns, severity, descriptions)
   - Trigger descriptions to broaden
   - Cross-skill integration rules to add

### Phase 4: Create PRs

10. For each skill repo with changes:
    - Clone the repo to a temp directory
    - Create a feature branch
    - Apply checkpoint/trigger changes
    - Commit with clear message referencing the session learnings
    - Push and create PR with structured description
    - Reply to any review comments
11. Track all created PRs

### Phase 5: Verify

12. Check CI passes on all PRs
13. Fix any review comments
14. Report summary of all PRs created

## Output Format

```
## Session Retrospective Summary

### Manual Work Performed
| # | Issue | Category | Should Have Been Caught By |
|---|-------|----------|---------------------------|

### Skill Gaps Found
| Skill | Gap Type | Fix |
|-------|----------|-----|

### PRs Created
| Repo | PR | New Checkpoints | Status |
|------|-----|-----------------|--------|

### Trigger Gaps
| Skill | Current Trigger | Missing Keywords | Fix |
|-------|----------------|------------------|-----|
```

## Key Principles

- **Skills improve at their source** — PRs go to the skill's GitHub repo, not local config
- **Checkpoints catch, skills instruct** — mechanical checks for what CAN be automated, skill docs for workflow guidance
- **Assessment before manual work** — every skill that does quality enhancement should instruct running automated-assessment first
- **Agent-harness is the enforcer** — it must verify that quality delegation actually produces results
- **Team benefits, not just one user** — memory is local, skill improvements are org-wide

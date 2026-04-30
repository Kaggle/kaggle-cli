# Kaggle Benchmarks Agent Skill

This directory contains an **Agent Skill** — a structured markdown document that teaches AI coding assistants how to use the `kaggle` CLI for benchmark workflows (push, run, status, download).

## Files

- **`SKILL.md`** — Full CLI reference with command syntax, examples, error messages, and workflow recipes.

## Compatibility

The skill body is plain markdown. The YAML frontmatter (`name`, `description`) is used by Gemini CLI for auto-discovery and is safely ignored by other tools.

## Setup by Agent

### Gemini CLI

Works natively — no setup needed. Gemini auto-discovers skills in `.gemini/skills/` and activates them on-demand when a matching request is detected.

```
# To manage skills manually in an interactive session:
/skills list
/skills enable kaggle-benchmarks
```

### Other Agents

Since `SKILL.md` is plain markdown, any AI coding assistant can use it. The general approach is the same for all tools — add the file to the agent's context:

| Agent | Approach |
|-------|----------|
| **Claude Code** | Reference the file path in your project's `CLAUDE.md`, or copy it into `.claude/commands/` as an on-demand command |
| **Cursor** | Reference the file path in `.cursorrules` or add a rule in `.cursor/rules/` |
| **GitHub Copilot** | Reference or inline in `.github/copilot-instructions.md`, or use `@workspace` in chat to point at the file |
| **Aider** | Use `/read .gemini/skills/kaggle-benchmarks/SKILL.md` to add it to the session context |
| **Any other agent** | Point the agent to `.gemini/skills/kaggle-benchmarks/SKILL.md` and ask it to read the file |

> **Tip:** Each tool has its own conventions for persistent instructions vs. on-demand context. Check your agent's documentation for the recommended way to include project-specific reference files.


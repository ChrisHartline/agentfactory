# Local Setup Guide

This project uses **Claude Code** and **Cursor** with local configuration files that contain machine-specific paths. If you clone this repo, you'll need to update these before things work.

---

## 1. Claude Code — `.claude/settings.local.json`

Claude Code uses this file to pre-approve shell commands so you aren't prompted every time. The current file contains **absolute paths** tied to the original developer's machine.

### What's in it

```json
{
  "permissions": {
    "allow": [
      "Bash(chmod:*)",
      "Bash(/Users/christopherhartline/Desktop/Development/agent-knowledge/scripts/generate_cursor_rule.sh:*)",
      "Bash(/Users/christopherhartline/Desktop/Development/agent-knowledge/scripts/sync_to_project.sh:*)",
      "Bash(python -m mcp_tools.servers.calculator:*)",
      "Bash(python:*)"
    ]
  }
}
```

### What to change

Replace the absolute paths with your own. The pattern is:

```
/Users/<your-username>/Desktop/Development/agent-knowledge/scripts/...
```

Or, if you put the knowledge base somewhere else, point to wherever your `agent-knowledge/scripts/` directory lives.

| Permission | Why it exists |
|---|---|
| `chmod:*` | Making knowledge base scripts executable |
| `generate_cursor_rule.sh:*` | Converting SKILL.md → CURSOR.mdc |
| `sync_to_project.sh:*` | Symlinking knowledge units into projects |
| `python -m mcp_tools.servers.calculator:*` | Running MCP tool servers for testing |
| `python:*` | General Python execution (demo scripts, testing) |

### If you don't use Claude Code

Delete the `.claude/` directory. It has no effect on Cursor or the Python code.

---

## 2. Cursor Rules — `.cursor/rules/`

Cursor Rules (`.mdc` files) give Cursor context about the project. This project has **four rules**, three of which are **symlinks** to a shared knowledge base.

### Current state

```
.cursor/rules/
├── sara-agent.mdc          ← Regular file (committed, works everywhere)
├── ops-planning.mdc        → ~/Desktop/Development/agent-knowledge/units/ops-planning/CURSOR.mdc
├── factory-dev.mdc         → ~/Desktop/Development/agent-knowledge/units/factory-dev/CURSOR.mdc
└── agent-creator.mdc       → ~/Desktop/Development/agent-knowledge/units/agent-creator/CURSOR.mdc
```

### The symlink problem

The three symlinked `.mdc` files point to absolute paths on the original machine. If you clone this repo:

- **The symlinks will be broken** (targets won't exist on your machine)
- Cursor will ignore broken symlinks silently — no errors, just missing context
- `sara-agent.mdc` will still work (it's a regular file)

### How to fix it

**Option A: Set up the shared knowledge base (recommended)**

This gives you the full knowledge unit system with self-improvement loop support.

```bash
# 1. Create the knowledge base (sibling to agent_factory/)
mkdir -p ~/Desktop/Development/agent-knowledge/{units,scripts,loop}

# 2. Copy the unit source files from this repo
#    (they're the source of truth for the .mdc symlinks)
cp -r /path/to/agent_factory/knowledge-units/* ~/Desktop/Development/agent-knowledge/units/

# 3. Make scripts executable
chmod +x ~/Desktop/Development/agent-knowledge/scripts/*.sh

# 4. Generate CURSOR.mdc files from SKILL.md
~/Desktop/Development/agent-knowledge/scripts/generate_cursor_rule.sh

# 5. Re-create symlinks pointing to YOUR knowledge base
~/Desktop/Development/agent-knowledge/scripts/sync_to_project.sh \
    ~/Desktop/Development/agent_factory \
    ops-planning factory-dev agent-creator
```

**Option B: Convert symlinks to regular files (simpler)**

If you just want the rules to work without the knowledge base infrastructure:

```bash
cd /path/to/agent_factory/.cursor/rules/

# Remove broken symlinks
rm ops-planning.mdc factory-dev.mdc agent-creator.mdc

# Copy the content directly (if you have the knowledge base source)
# Or just use the project without those rules — sara-agent.mdc
# provides the core project context.
```

### What each rule does

| Rule | Type | Purpose |
|---|---|---|
| `sara-agent.mdc` | File | Project identity for Cursor — defines Sara as the AI engineer, includes Linear integration IDs, team ecosystem awareness, and technical standards |
| `ops-planning.mdc` | Symlink | OPORD-based planning skill — activates when tasks have 3+ subtasks or need decomposition before execution |
| `factory-dev.mdc` | Symlink | Factory development skill — project file map, template creation checklist, testing patterns, architecture decisions |
| `agent-creator.mdc` | Symlink | Meta-agent skill — how to search the registry, create new agent templates, spawn and test agents |

### Note on `sara-agent.mdc`

This file contains **Linear project IDs** specific to the Sacrumcor team. If you're forking this project, you'll want to either:
- Replace the Linear section with your own project management integration
- Remove the Linear section entirely (the rest of the file still works)

---

## 3. Shared Knowledge Base

The symlinks above point to a **shared knowledge base** that lives outside this repo:

```
~/Desktop/Development/
├── agent-knowledge/          ← Shared knowledge base (NOT in this repo)
│   ├── units/                ← SKILL.md + generated CURSOR.mdc per unit
│   ├── scripts/              ← generate_cursor_rule.sh, sync_to_project.sh
│   └── loop/                 ← AAR template for self-improvement
│
├── agent_factory/            ← This repo
│   └── .cursor/rules/        ← Symlinks point back to agent-knowledge/
│
└── [other projects]/         ← Can also symlink to same knowledge units
```

The knowledge base is designed to be **shared across projects**. When you update a SKILL.md in the knowledge base, every project that symlinks to it gets the update automatically.

See `agent-knowledge-schema.md` in this repo for the full schema specification.

---

## Quick Start Checklist

- [ ] Clone the repo
- [ ] `cp .env.example .env` and add your API keys
- [ ] `pip install -r requirements.txt`
- [ ] Update paths in `.claude/settings.local.json` (or delete `.claude/` if not using Claude Code)
- [ ] Fix symlinks in `.cursor/rules/` using Option A or B above (or ignore if not using Cursor)
- [ ] `python demo_factory.py` to verify everything loads

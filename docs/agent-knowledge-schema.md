# Agent Knowledge Unit Schema
## A Portable Skill/Rule Format for Recursive Self-Improvement

Version: 0.1.0-draft
Status: Working draft — designed to be used, broken, and revised

---

## 1. DESIGN GOALS

- **Portable** across Claude Skills and Cursor Rules with minimal adaptation
- **Self-improving** — each unit carries its own evaluation criteria so agents can assess and propose revisions
- **Composable** — units can reference and depend on each other
- **Auditable** — version history tracks what changed, why, and what triggered the change

---

## 2. DIRECTORY STRUCTURE

A single knowledge unit lives in its own directory:

```
unit-name/
├── SKILL.md              # Primary instruction file (Claude Skills compatible)
├── CURSOR.mdc            # Generated/symlinked adapter for Cursor Rules (optional)
├── references/           # Supporting docs loaded on demand
├── scripts/              # Deterministic code the agent can execute
├── assets/               # Templates, images, etc. used in output
└── .history/             # Version snapshots (gitignored or separate)
    ├── v0.1.0.md
    └── changelog.md
```

### Compatibility Notes

- `SKILL.md` is the source of truth. It follows Claude's existing skill conventions.
- `CURSOR.mdc` is a thin adapter generated from SKILL.md (see Section 5).
- Both systems consume Markdown with YAML frontmatter, so the body content is shared.

---

## 3. SKILL.md SCHEMA

```yaml
---
# === IDENTITY (required) ===
name: unit-name
description: >
  What this unit does and when to trigger it. This is the primary
  matching text — be specific about trigger conditions.
  (Same conventions as standard Claude Skills.)

# === SELF-IMPROVEMENT METADATA (new fields, ignored by Claude runtime) ===
version: 0.1.0
last_updated: 2026-02-06
update_source: manual | agent-proposed | aar-derived
domain: navigation | medical | coding | ops-planning | general
depends_on: []            # other unit names this unit assumes are available
---
```

### Body Sections

The Markdown body uses the following sections. Not all are required for every unit — use what fits. The key addition over standard Claude Skills is **Sections E and F**, which enable the self-improvement loop.

```markdown
# Unit Name

## A. Purpose & Trigger Conditions
When should this unit activate? What task patterns match?
Include positive examples (use this when...) and negative examples (don't use for...).

## B. Prerequisites
What must be true before executing? Other units loaded, tools available,
files present, environment state, etc.

## C. Procedures
The core instructions. This is the "how to do the work" section.
Use the degree-of-freedom principle:
  - High freedom: heuristics, guidelines
  - Medium freedom: pseudocode, decision trees
  - Low freedom: exact scripts, specific sequences

Subsections as needed (C.1, C.2, etc.)

## D. Resources
Pointers to bundled references, scripts, and assets.
Brief description of each and when to load it.

## E. Evaluation Criteria
How does the agent (or human) assess whether this unit performed well?

### E.1 Output Quality Indicators
- What does "good" look like? Concrete, observable markers.
- What are common failure modes?

### E.2 Efficiency Indicators
- Expected tool call range (e.g., "should complete in 2-4 tool calls")
- Red flags that suggest the procedure is wrong

### E.3 Self-Assessment Prompt
A specific prompt the agent can use after task completion to evaluate
its own performance against criteria above. Example:

> After completing this task, assess: (1) Did the output meet the quality
> indicators in E.1? (2) Were there unnecessary steps? (3) What would you
> change in the procedure for next time?

## F. Revision History
| Version | Date       | Source         | Change Summary                    |
|---------|------------|----------------|-----------------------------------|
| 0.1.0   | 2026-02-06 | manual         | Initial draft                     |
```

---

## 4. THE OPORD PARALLEL

The schema's body sections map to the five-paragraph OPORD structure.
This isn't decorative — it means any agent trained on military planning
doctrine already has an intuition for how to read and use these units.

```
OPORD Paragraph         →  Schema Section
─────────────────────────────────────────────
1. Situation            →  A. Purpose & Trigger (what's the context?)
2. Mission              →  A. Purpose (what's the objective?)
3. Execution            →  C. Procedures (how do we accomplish it?)
4. Sustainment          →  B. Prerequisites + D. Resources (what do we need?)
5. Command & Signal     →  E. Evaluation Criteria (how do we know we're done?)
                           F. Revision History (the AAR feeds back here)
```

The AAR (After Action Review) is the mechanism that closes the loop:
execution results feed back into Situation for the next cycle.

---

## 5. CURSOR ADAPTER GENERATION

A Cursor `.mdc` file needs slightly different frontmatter. Rather than
maintaining two files, generate `CURSOR.mdc` from `SKILL.md`:

```bash
#!/bin/bash
# generate_cursor_rule.sh — converts SKILL.md to CURSOR.mdc
# Usage: ./generate_cursor_rule.sh path/to/unit-name

UNIT_DIR="$1"
SKILL_FILE="$UNIT_DIR/SKILL.md"
OUTPUT="$UNIT_DIR/CURSOR.mdc"

if [ ! -f "$SKILL_FILE" ]; then
  echo "Error: $SKILL_FILE not found"
  exit 1
fi

# Extract fields from YAML frontmatter
NAME=$(grep '^name:' "$SKILL_FILE" | head -1 | sed 's/name: *//')
DESC=$(grep '^description:' "$SKILL_FILE" | head -1 | sed 's/description: *//')

# Extract body (everything after second ---)
BODY=$(awk '/^---$/{n++} n==2{if(n==2 && /^---$/) next; print}' "$SKILL_FILE")

# Write Cursor .mdc format
cat > "$OUTPUT" << EOF
---
description: $DESC
globs:
alwaysApply: false
---

$BODY
EOF

echo "Generated $OUTPUT"
```

### Symlink Strategy

```
~/agent-knowledge/
├── units/
│   ├── protobuf-patterns/
│   │   ├── SKILL.md
│   │   └── CURSOR.mdc        # generated
│   ├── ekf-fusion/
│   │   ├── SKILL.md
│   │   └── CURSOR.mdc
│   └── ops-planning/
│       ├── SKILL.md
│       └── CURSOR.mdc
├── scripts/
│   ├── generate_cursor_rule.sh
│   ├── sync_to_claude.sh      # copies/symlinks to Claude skill dirs
│   └── sync_to_cursor.sh      # symlinks .mdc into project .cursor/rules/
└── loop/
    └── aar-template.md        # AAR template for step 4 of the loop
```

Sync scripts create symlinks per-project:

```bash
# sync_to_cursor.sh — link relevant units into a project
# Usage: ./sync_to_cursor.sh ~/projects/falcon ekf-fusion protobuf-patterns

PROJECT="$1"; shift
RULES_DIR="$PROJECT/.cursor/rules"
mkdir -p "$RULES_DIR"

for UNIT in "$@"; do
  SRC="$HOME/agent-knowledge/units/$UNIT/CURSOR.mdc"
  DEST="$RULES_DIR/$UNIT.mdc"
  if [ -f "$SRC" ]; then
    ln -sf "$SRC" "$DEST"
    echo "Linked $UNIT → $DEST"
  else
    echo "Warning: $SRC not found. Run generate_cursor_rule.sh first."
  fi
done
```

---

## 6. THE SELF-IMPROVEMENT LOOP

Six steps. The agent executes steps 1-3 autonomously.
Step 4 is the human gate. Steps 5-6 are propagation.

```
    ┌──────────────────────────────────────────────┐
    │                                              │
    ▼                                              │
┌─────────┐   ┌─────────┐   ┌─────────┐          │
│ 1.TASK  │──▶│ 2.EVAL  │──▶│3.PROPOSE│          │
│ Execute │   │ Assess  │   │ Revise  │          │
└─────────┘   └─────────┘   └─────────┘          │
                                 │                 │
                                 ▼                 │
                           ┌─────────┐            │
                           │4.REVIEW │            │
                           │ Human   │            │
                           │  Gate   │            │
                           └─────────┘            │
                            │       │              │
                         reject   approve          │
                            │       │              │
                            │       ▼              │
                            │  ┌─────────┐        │
                            │  │5.UPDATE │        │
                            │  │ Commit  │        │
                            │  └─────────┘        │
                            │       │              │
                            │       ▼              │
                            │  ┌─────────┐        │
                            │  │6.SPREAD │────────┘
                            │  │Propagate│
                            │  └─────────┘
                            │
                            └──▶ (back to 1, unit unchanged)
```

### Step 1: TASK — Execute Work Using Current Units

The agent receives a task, identifies relevant knowledge units,
reads their SKILL.md files, and executes the procedures.

**Input:** User request + loaded knowledge units
**Output:** Task deliverable + execution trace (what units were used, what tools called)

### Step 2: EVAL — Assess Against Evaluation Criteria

The agent runs the Self-Assessment Prompt from Section E.3 of
each unit that was used. This produces a structured assessment.

**Input:** Completed deliverable + unit evaluation criteria (Section E)
**Output:** Assessment document with scores/observations per criterion

Template:
```markdown
## Post-Task Assessment
- **Unit used:** [unit-name] v[version]
- **Task:** [brief description]
- **Quality indicators met:** [yes/partial/no per E.1 item]
- **Efficiency:** [tool calls used vs expected range from E.2]
- **Failure modes observed:** [any from E.1 list, or new ones]
- **Proposed changes:** [specific edits to Sections A-E]
- **Confidence:** [low/medium/high that proposed changes improve outcomes]
```

### Step 3: PROPOSE — Draft Unit Revision

If the assessment identifies improvements, the agent drafts a
revised SKILL.md with:
- Specific diff (what changed in which section)
- Rationale tied to assessment observations
- Updated version number (semver: patch for tweaks, minor for new content)
- New entry in Section F revision history

**Key constraint:** The agent proposes, it does not commit.

### Step 4: REVIEW — Human Gate

The human reviews the proposed revision. This is the AAR.

Questions for the reviewer:
1. Does the proposed change match your experience of the task outcome?
2. Is the change scoped correctly (not over-generalizing from one instance)?
3. Any unintended consequences for other units that depend on this one?

**Outcomes:** approve, reject, or modify-then-approve.

### Step 5: UPDATE — Commit the Revision

On approval:
- Current SKILL.md snapshot saved to `.history/v[old-version].md`
- New SKILL.md written with updated content and version
- Changelog updated
- CURSOR.mdc regenerated via `generate_cursor_rule.sh`

### Step 6: SPREAD — Propagate via Symlinks

Because units are symlinked into projects, the update is
immediately available everywhere the unit is linked.

- Verify symlinks are intact
- If the unit has dependents, flag them for review
- Next task execution (back to Step 1) uses the updated unit

---

## 7. GETTING STARTED

### First unit in 5 minutes:

```bash
# 1. Create the knowledge base
mkdir -p ~/agent-knowledge/{units,scripts,loop}

# 2. Create your first unit
mkdir -p ~/agent-knowledge/units/my-first-unit

# 3. Write a minimal SKILL.md (copy the schema from Section 3)
#    Start with just Sections A and C. Add E when you're ready
#    to start the improvement loop.

# 4. Save the sync scripts from Section 5

# 5. Link it into a project
./sync_to_cursor.sh ~/projects/my-project my-first-unit
```

### Suggested first units for Chris:

Based on existing work patterns:
- `ekf-fusion` — EKF threading and sensor fusion procedures for Jetson
- `protobuf-dds` — Protobuf + DDS middleware patterns for FALCON
- `r-preprocessing` — Data preprocessing techniques for ML coursework
- `ops-planning` — OPORD/JOPP structured planning for agent tasking

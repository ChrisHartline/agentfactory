---
name: ops-planning
description: >
  Use this unit when the agent receives a complex, multi-step task that
  benefits from structured planning before execution. Triggers include:
  tasks with multiple deliverables, ambiguous requirements needing
  decomposition, tasks requiring coordination across tools or domains,
  or any request where jumping straight to execution would likely require
  significant rework. Do NOT use for simple, single-step requests.
version: 0.1.0
last_updated: 2026-02-06
update_source: manual
domain: ops-planning
depends_on: []
---

# Ops Planning — OPORD-Based Agent Tasking

## A. Purpose & Trigger Conditions

**Use when:**
- Task has 3+ distinct subtasks or deliverables
- Requirements are ambiguous and need decomposition before execution
- Multiple tools, files, or domains are involved
- Failure cost is high (e.g., generating a document the user will send externally)

**Don't use for:**
- Simple Q&A or factual lookups
- Single-file edits with clear instructions
- Conversational exchanges

## B. Prerequisites

- No other units required
- This unit is a planning wrapper — it operates *before* domain-specific units activate

## C. Procedures

### C.1 Receive and Frame (OPORD Paragraph 1: Situation)

Before acting, extract or infer the following from the user's request:

- **Context**: What do we already know? Prior conversation, uploaded files, stated constraints.
- **Environment**: What tools are available? What are the platform limitations?
- **Assumptions**: What are we assuming that hasn't been explicitly stated? Flag these.

If critical context is missing, ask — but limit to one focused clarifying question.

### C.2 Define the Mission (OPORD Paragraph 2: Mission)

State the task as a single sentence with:
- **WHO**: Which agent/tool is responsible
- **WHAT**: The specific deliverable(s)
- **WHEN**: Any time/sequence constraints
- **WHY**: The user's underlying intent (not just the literal request)

Example: "Create a formatted .docx report (WHAT) using the docx skill (WHO) that synthesizes the uploaded CSV data into executive-readable findings (WHY), completing the analysis before formatting (WHEN)."

### C.3 Plan Execution (OPORD Paragraph 3: Execution)

Break the mission into phases. For each phase:
- **Action**: What specifically happens
- **Tool/Unit**: What skill or tool is used
- **Input**: What this phase consumes
- **Output**: What this phase produces (becomes input to next phase)
- **Decision point**: Under what conditions do we proceed vs. pause for feedback

Keep to 3-5 phases. If you need more, you're probably bundling multiple missions.

### C.4 Identify Resources (OPORD Paragraph 4: Sustainment)

Inventory what's needed:
- Files to read or create
- Skills/units to load
- External tools or APIs
- Information gaps that require web search or user input

### C.5 Define Coordination (OPORD Paragraph 5: Command & Signal)

Specify:
- How progress will be communicated to the user (interim updates vs. final delivery)
- What triggers escalation (e.g., "if the data has >20% missing values, ask before imputing")
- How the final output will be delivered

### C.6 Execute and Adapt

Execute the plan. After each phase:
- Verify the output meets the phase's expected output
- If it doesn't, adapt — revise remaining phases, don't restart from scratch
- If a blocking issue emerges, surface it to the user with a proposed resolution

## D. Resources

None currently bundled. This is a pure-procedural unit.

Future: could include planning templates, phase-checklist scripts.

## E. Evaluation Criteria

### E.1 Output Quality Indicators
- Plan was stated before execution began (not implicit)
- Each phase produced a usable input for the next phase
- Final deliverable matched user intent, not just literal request
- No unnecessary rework or backtracking

**Common failure modes:**
- Jumping to execution without planning (most frequent)
- Over-planning simple tasks (applying this unit when it wasn't needed)
- Plan too rigid — didn't adapt when intermediate output was unexpected

### E.2 Efficiency Indicators
- Planning overhead should be <15% of total tool calls
- If planning takes more than 2 tool calls, the plan is too complex

### E.3 Self-Assessment Prompt

> After completing this task, assess:
> 1. Did I state a plan before executing? Was it visible to the user?
> 2. Did any phase produce unexpected output that required replanning?
> 3. Was the planning overhead proportional to task complexity?
> 4. Would a different phase breakdown have been more efficient?
> 5. What would I add to Section C to handle this task better next time?

## F. Revision History

| Version | Date       | Source | Change Summary |
|---------|------------|--------|----------------|
| 0.1.0   | 2026-02-06 | manual | Initial draft  |

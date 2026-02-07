"""
Prompt Composer — assembles system prompts from templates + tool instructions.

Replaces {{TOOL_BLOCK:tool_id}} placeholders with actual tool descriptions,
optionally injects meta-reasoning, and appends task context.
"""

from __future__ import annotations

import re

from .models import (
    PromptTemplate,
    ReasoningFramework,
    SpawnConfig,
)
from .registry import ToolRegistry


SIRP_INJECTION = """
# Meta-Reasoning: Structured Iterative Reasoning Protocol (SIRP)

Apply this self-monitoring discipline throughout your work:

Enclose exploratory thoughts in <thinking> tags before committing to an approach.
Track your step budget with <count> tags (starting budget: {step_budget}).
After every 3-4 steps, pause for <reflection>:
  - Score your progress 0.0-1.0
  - If quality > 0.8: continue current approach
  - If quality 0.5-0.8: adjust strategy
  - If quality < 0.5: backtrack and try a different approach

SIRP tags are for internal reasoning. Your final output should
match the Output Contract specified above.
"""

STANDARD_REFLECTION = """
# Self-Reflection

After completing your analysis, briefly assess:
- Did I address the core question?
- What is my confidence in this answer?
- What would I do differently with more time/data?
"""


class PromptComposer:
    """
    Assembles a complete system prompt from template + tool instructions.

    Composition:
    1. Base template (identity, reasoning protocol, contract, guardrails)
    2. Tool blocks injected via {{TOOL_BLOCK:tool_id}} replacement
    3. Optional meta-reasoning (SIRP or standard reflection)
    4. Task context from the orchestrator
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def compose(self, template: PromptTemplate, config: SpawnConfig) -> str:
        prompt = template.system_prompt

        # 1. Inject tool blocks
        prompt = self._inject_tool_blocks(prompt, template, config)

        # 2. Meta-reasoning (optional)
        framework = config.reasoning_framework or template.reasoning_framework
        if framework == ReasoningFramework.SIRP:
            budget = config.max_iterations or template.max_iterations or 20
            prompt += SIRP_INJECTION.format(step_budget=budget)
        elif framework == ReasoningFramework.STANDARD:
            prompt += STANDARD_REFLECTION

        # 3. Task context
        if config.task_context:
            prompt += f"\n\n# Current Task\n\n{config.task_context}\n"

        return prompt

    def _inject_tool_blocks(
        self,
        prompt: str,
        template: PromptTemplate,
        config: SpawnConfig,
    ) -> str:
        """Replace {{TOOL_BLOCK:tool_id}} placeholders with tool instructions."""
        if config.tool_overrides is not None:
            tool_ids = config.tool_overrides
        else:
            tool_ids = template.required_tools + template.optional_tools

        pattern = r"\{\{TOOL_BLOCK:(\w+)\}\}"
        found_blocks = re.findall(pattern, prompt)

        for tool_id in found_blocks:
            placeholder = f"{{{{TOOL_BLOCK:{tool_id}}}}}"

            if tool_id in tool_ids:
                instructions = self.tool_registry.get_prompt_instructions(tool_id)
                if instructions:
                    prompt = prompt.replace(placeholder, instructions)
                elif tool_id in template.required_tools:
                    prompt = prompt.replace(
                        placeholder,
                        f"## Tool: {tool_id}\n(Available — use standard calling conventions.)",
                    )
                else:
                    prompt = prompt.replace(placeholder, "")
            else:
                if tool_id in template.required_tools:
                    prompt = prompt.replace(
                        placeholder,
                        f"## Tool: {tool_id} — NOT AVAILABLE\nAdapt your approach accordingly.",
                    )
                else:
                    prompt = prompt.replace(placeholder, "")

        # Clean up triple+ blank lines
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)
        return prompt

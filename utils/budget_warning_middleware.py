"""Budget warning middleware - progressive budget warnings to prevent Agent from looping.

Injects progressive warning messages into the conversation via wrap_model_call hook
when the Agent approaches the recursion_limit, prompting it to summarize and output
a final answer before being hard-interrupted with no output.

Additionally, provides a "force answer" fallback: when remaining_steps is extremely low
(below force_answer_threshold), the middleware sets tool_choice="none" and clears tools,
making it physically impossible for the model to call tools. This guarantees a text
response and prevents GraphRecursionError.

Uses wrap_model_call instead of before_model to avoid consuming an extra graph step
per tool call iteration. wrap_model_call executes inside the model node, so it does
NOT add an extra node to the graph loop.

Uses LangGraph's RemainingSteps managed value to directly read remaining steps
without consuming any extra recursion_limit quota (read-only, no side effects).
"""

import sys
from typing import Any, Annotated, Callable, Awaitable

from langchain_core.messages import HumanMessage
from langgraph.managed.is_last_step import RemainingStepsManager
from langgraph.typing import ContextT

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)


# Unique prefix used to identify injected warnings in messages (for dedup)
_WARNING_MARKER = "[SYSTEM BUDGET WARNING"

# Default force answer threshold: when remaining_steps <= this value,
# tools are disabled and the model is forced to output a text answer.
# 4 steps = enough for model(1) + after_model(1) + end routing(~2)
DEFAULT_FORCE_ANSWER_THRESHOLD = 4

_FORCE_ANSWER_MESSAGE = (
    "[SYSTEM BUDGET WARNING - FINAL] This is your LAST model call. "
    "All tools have been disabled. You MUST output your final answer NOW.\n"
    "- Synthesize all evidence gathered so far\n"
    "- Give your best answer based on available information\n"
    "- If evidence conflicts, go with the majority\n"
    "- Output ONLY the final answer, no tool calls"
)

# Default 3-level warning thresholds: (percentage_used, severity, message_template)
# Triggers when steps used reaches recursion_limit * percentage_used
# (i.e., remaining <= limit * (1 - percentage)).
DEFAULT_WARNING_THRESHOLDS = [
    (0.65, "NOTICE", (
        "[SYSTEM BUDGET WARNING] You have approximately {remaining_calls} tool calls remaining. "
        "Be mindful of your budget:\n"
        "- Avoid redundant verification of already-established facts\n"
        "- Each remaining call should gather genuinely NEW information\n"
        "- If you have consistent evidence from 3+ sources, commit to that answer"
    )),
    (0.80, "WARNING", (
        "[SYSTEM BUDGET WARNING] You have approximately {remaining_calls} tool calls remaining. "
        "You MUST start wrapping up:\n"
        "- If you have unresolved contradictions, apply majority vote NOW\n"
        "- Do NOT continue verifying what you already know\n"
        "- Begin synthesizing your findings into a final answer"
    )),
    (0.90, "CRITICAL", (
        "[SYSTEM BUDGET WARNING - CRITICAL] You have approximately {remaining_calls} tool calls remaining. "
        "You MUST output your final answer IMMEDIATELY.\n"
        "- Do NOT make any more verification or exploration calls\n"
        "- Summarize all evidence you have gathered\n"
        "- Give your best answer based on available information, even if uncertain\n"
        "- If evidence conflicts, go with the majority and note the uncertainty"
    )),
]


class BudgetWarningState(AgentState[ResponseT]):
    """State schema for BudgetWarningMiddleware.

    Extends AgentState with:
    - remaining_steps: LangGraph managed value, auto-injected remaining graph step count
    """
    remaining_steps: Annotated[int, PrivateStateAttr, RemainingStepsManager]


class BudgetWarningMiddleware(AgentMiddleware[BudgetWarningState[ResponseT], ContextT, ResponseT]):
    """Injects progressive warning messages when the Agent approaches recursion_limit.

    Uses wrap_model_call hook (executes inside model node, no extra graph step) to read
    state["remaining_steps"] and inject warning HumanMessage into request.messages before
    the LLM call.

    Deduplication: counts existing warnings in messages (by _WARNING_MARKER prefix) to
    determine current warning level, ensuring each level fires only once.

    Thresholds are dynamically computed from recursion_limit * percentage:
    - 65% used: gentle reminder
    - 80% used: force wrap-up
    - 90% used: critical, must output answer immediately

    Example with recursion_limit=125 (3 steps per tool call -> ~42 tool calls total):
    - remaining <= 125*(1-0.65)=43 -> NOTICE, ~14 tool calls left
    - remaining <= 125*(1-0.80)=25 -> WARNING, ~8 tool calls left
    - remaining <= 125*(1-0.90)=12 -> CRITICAL, ~4 tool calls left
    """

    state_schema = BudgetWarningState  # type: ignore[assignment]

    def __init__(self, *, recursion_limit: int = 125, thresholds=None, force_answer_threshold: int = DEFAULT_FORCE_ANSWER_THRESHOLD):
        """Initialize the budget warning middleware.

        Args:
            recursion_limit: The actual recursion_limit value used to compute thresholds.
                             e.g., 125 as set in batch_worker.py / video-agent.py.
            thresholds: Custom threshold list of (percentage_used, severity, template) tuples.
                        percentage_used: float in [0, 1], fraction of limit already consumed.
                        Triggers when remaining_steps <= limit * (1 - percentage_used).
                        Defaults to DEFAULT_WARNING_THRESHOLDS (65%/80%/90%).
            force_answer_threshold: When remaining_steps <= this value, tools are disabled
                        and model is forced to output text. Default 4.
        """
        super().__init__()
        self.recursion_limit = recursion_limit
        self.thresholds = thresholds or DEFAULT_WARNING_THRESHOLDS
        self.force_answer_threshold = force_answer_threshold

    @property
    def name(self) -> str:
        return "BudgetWarningMiddleware"

    def _get_remaining_threshold(self, percentage_used: float) -> int:
        """Compute remaining_steps trigger threshold from percentage_used.

        Example: limit=125, percentage_used=0.65 -> threshold = 125 * (1 - 0.65) = 43
        """
        return int(self.recursion_limit * (1 - percentage_used))

    def _count_existing_warnings(self, messages: list) -> int:
        """Count how many budget warnings have already been injected in messages.

        This serves as the deduplication mechanism: instead of tracking state,
        we count existing warning messages to determine current level.
        """
        count = 0
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else ""
                if content.startswith(_WARNING_MARKER):
                    count += 1
        return count

    def _maybe_inject_warning(self, request: ModelRequest[ContextT]) -> None:
        """Check remaining steps and inject warning into request.messages if needed.

        Modifies request.messages in-place (appends HumanMessage warning).
        """
        remaining_steps = request.state.get("remaining_steps")
        if remaining_steps is None:
            return

        # Determine current warning level by counting existing warnings in messages
        current_level = self._count_existing_warnings(request.messages)

        # Iterate thresholds to determine which warning level to issue
        new_level = current_level
        warning_msg = None

        for level_idx, (percentage_used, severity, template) in enumerate(self.thresholds, 1):
            threshold = self._get_remaining_threshold(percentage_used)
            if remaining_steps <= threshold and current_level < level_idx:
                new_level = level_idx
                # Estimate remaining tool calls (each tool call ~ 3 graph steps:
                # model + Todo.after_model + tools)
                remaining_calls = max(1, remaining_steps // 3)
                warning_msg = template.format(
                    remaining_calls=remaining_calls,
                    remaining_steps=remaining_steps,
                )

        if warning_msg and new_level > current_level:
            print(f"\n[BudgetWarningMiddleware] Injecting level-{new_level} warning "
                  f"(remaining_steps={remaining_steps}/{self.recursion_limit})", file=sys.stderr)
            # Inject warning directly into request.messages before LLM sees them
            request.messages.append(HumanMessage(content=warning_msg))

    def _maybe_force_answer(self, request: ModelRequest[ContextT]) -> bool:
        """Force the model to output text when remaining steps are critically low.

        When remaining_steps <= force_answer_threshold:
        - Sets tool_choice to "none" (API-level enforcement)
        - Clears available tools
        - Injects FINAL warning message

        Returns True if force answer was triggered, False otherwise.
        """
        remaining_steps = request.state.get("remaining_steps")
        if remaining_steps is None:
            return False

        if remaining_steps <= self.force_answer_threshold:
            print(f"\n[BudgetWarningMiddleware] FORCE ANSWER triggered "
                  f"(remaining_steps={remaining_steps}/{self.recursion_limit}, "
                  f"threshold={self.force_answer_threshold}). "
                  f"Disabling tools, forcing text output.", file=sys.stderr)
            request.tool_choice = "none"
            request.tools = []
            request.messages.append(HumanMessage(content=_FORCE_ANSWER_MESSAGE))
            return True

        return False

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Wrap model call to inject budget warning or force answer before LLM execution.

        Executes inside the model node (no extra graph step consumed).
        Force answer takes priority over warning injection.
        """
        if not self._maybe_force_answer(request):
            self._maybe_inject_warning(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Async version of wrap_model_call."""
        if not self._maybe_force_answer(request):
            self._maybe_inject_warning(request)
        return await handler(request)

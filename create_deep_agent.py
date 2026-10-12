"""Deep Agents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from deepagents.middleware.summarization import create_summarization_middleware
from utils.tool_args_validator_middleware import ToolArgsValidatorMiddleware
from utils.budget_warning_middleware import BudgetWarningMiddleware
from utils.tool_selection import merge_disabled_builtin_tools

BASE_AGENT_PROMPT = \
"""You are a Deep Agent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls. The user can see your responses and tool outputs in real time.

## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble (\"Sure!\", \"Great question!\", \"I'll now...\").
- Don't say \"I'll now do X\" — just do it.
- If the request is ambiguous, ask questions before acting.
- If asked how to approach something, explain first, then act.

## Professional Objectivity

- Prioritize accuracy over validating the user's beliefs
- Disagree respectfully when the user is incorrect
- Avoid unnecessary superlatives, praise, or emotional validation

## Doing Tasks

When the user asks you to do something:

1. **Understand first** — read relevant files, check existing patterns. Quick but thorough — gather enough evidence to start, then iterate.
2. **Act** — implement the solution. Work quickly but accurately.
3. **Verify** — check your work against what was asked, not against your own output. Your first attempt is rarely correct — iterate.

Keep working until the task is fully complete. Don't stop partway and explain what you would do — just do it. Only yield back to the user when the task is done or you're genuinely blocked.

**When things go wrong:**
- If something fails repeatedly, stop and analyze *why* — don't keep retrying the same approach.
- If you're blocked, tell the user what's wrong and ask for guidance.

## Progress Updates

For longer tasks, provide brief progress updates at reasonable intervals — a concise sentence recapping what you've done and what's next."""  # noqa: E501


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        `ChatAnthropic` instance configured with Claude Sonnet 4.6.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-6",
    )


def resolve_model(model: str | BaseChatModel) -> BaseChatModel:
    """Resolve a model string to a `BaseChatModel` instance.

    If `model` is already a `BaseChatModel`, returns it unchanged.

    String models are resolved via `init_chat_model`, with OpenAI models
    defaulting to the Responses API. See the `create_deep_agent` docstring for
    details on how to customize this behavior.

    Args:
        model: Model name string or pre-configured model instance.

    Returns:
        Resolved `BaseChatModel` instance.
    """
    if isinstance(model, BaseChatModel):
        return model
    if model.startswith("openai:"):
        # Use Responses API by default. To use chat completions, use
        # `model=init_chat_model("openai:...")`
        # To disable data retention with the Responses API, use
        # `model=init_chat_model("openai:...", use_responses_api=True, store=False, include=["reasoning.encrypted_content"])`
        return init_chat_model(model, use_responses_api=True)
    return init_chat_model(model)


def _filter_middleware_tools(
    middleware_list: list,
    disabled_tools: set[str],
    remove_empty_middleware: bool = True
) -> list[str]:
    """Remove disabled tools from middleware in place.

    Args:
        middleware_list: List of middleware instances.
        disabled_tools: Set of tool names to disable.
        remove_empty_middleware: Whether to remove a middleware instance when all its tools are removed.

    Returns:
        Class names of middleware instances removed entirely.
    """
    removed_middleware_names = []
    to_remove = []

    for i, mw in enumerate(middleware_list):
        tools = getattr(mw, "tools", None)
        if not tools:
            continue

        old_tools = list(tools)
        removed_names = [
            getattr(t, "name", "")
            for t in old_tools
            if getattr(t, "name", "") in disabled_tools
        ]

        if removed_names:
            new_tools = [
                t for t in old_tools
                if getattr(t, "name", "") not in disabled_tools
            ]

            if not new_tools and remove_empty_middleware:
                # All tools are disabled; mark the middleware instance for removal
                to_remove.append(i)
                removed_middleware_names.append(mw.__class__.__name__)
                # print(f"🗑️ {mw.__class__.__name__} All tools are disabled; removing this middleware instance")
            else:
                mw.tools = new_tools
                print(f"⚠️ Disabled tools in {mw.__class__.__name__}: {removed_names}")

    # Remove in reverse order to keep indices valid
    for i in reversed(to_remove):
        mw = middleware_list[i]
        print(f"🗑️ {mw.__class__.__name__} All tools are disabled; removing this middleware instance")
        del middleware_list[i]

    return removed_middleware_names

def create_deep_agent(  # noqa: C901, PLR0912  # Complex graph assembly logic with many conditional branches
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    disabled_builtin_tools: list[str] | None = None,
    recursion_limit: int = 125,
    enable_budget_warning: bool = True,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """Create a deep agent.

    !!! warning "Deep agents require a LLM that supports tool calling!"

    By default, this agent has access to the following tools:

    - `write_todos`: manage a todo list
    - `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`: file operations
    - `execute`: run shell commands
    - `task`: call subagents

    The `execute` tool allows running shell commands if the backend implements `SandboxBackendProtocol`.
    For non-sandbox backends, the `execute` tool will return an error message.

    Args:
        model: The model to use.

            Defaults to `claude-sonnet-4-6`.

            Use the `provider:model` format (e.g., `openai:gpt-5`) to quickly switch between models.

            If an `openai:` model is used, the agent will use the OpenAI
            Responses API by default. To use OpenAI chat completions instead,
            initialize the model with
            `init_chat_model("openai:...", use_responses_api=False)` and pass
            the initialized model instance here. To disable data retention with
            the Responses API, use
            `init_chat_model("openai:...", use_responses_api=True, store=False, include=["reasoning.encrypted_content"])`
            and pass the initialized model instance here.
        tools: The tools the agent should have access to.

            In addition to custom tools you provide, deep agents include built-in tools for planning,
            file management, and subagent spawning.
        system_prompt: Custom system instructions to prepend before the base deep agent
            prompt.

            If a string, it's concatenated with the base prompt.
        middleware: Additional middleware to apply after the standard middleware stack
            (`TodoListMiddleware`, `FilesystemMiddleware`, `SubAgentMiddleware`,
            `SummarizationMiddleware`, `AnthropicPromptCachingMiddleware`,
            `PatchToolCallsMiddleware`).
        subagents: The subagents to use.

            Each subagent should be a `dict` with the following keys:

            - `name`
            - `description` (used by the main agent to decide whether to call the sub agent)
            - `system_prompt` (used as the system prompt in the subagent)
            - (optional) `tools`
            - (optional) `model` (either a `LanguageModelLike` instance or `dict` settings)
            - (optional) `middleware` (list of `AgentMiddleware`)
        skills: Optional list of skill source paths (e.g., `["/skills/user/", "/skills/project/"]`).

            Paths must be specified using POSIX conventions (forward slashes) and are relative
            to the backend's root. When using `StateBackend` (default), provide skill files via
            `invoke(files={...})`. With `FilesystemBackend`, skills are loaded from disk relative
            to the backend's `root_dir`. Later sources override earlier ones for skills with the
            same name (last one wins).
        memory: Optional list of memory file paths (`AGENTS.md` files) to load
            (e.g., `["/memory/AGENTS.md"]`).

            Display names are automatically derived from paths.

            Memory is loaded at agent startup and added into the system prompt.
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional `Checkpointer` for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses `StoreBackend`).
        backend: Optional backend for file storage and execution.

            Pass either a `Backend` instance or a callable factory like `lambda rt: StateBackend(rt)`.
            For execution support, use a backend that implements `SandboxBackendProtocol`.
        interrupt_on: Mapping of tool names to interrupt configs.

            Pass to pause agent execution at specified tool calls for human approval or modification.

            Example: `interrupt_on={"edit_file": True}` pauses before every edit.
        disabled_builtin_tools: List of built-in tool names to disable.
            Applied to the main agent and subagents constructed here. Explicit
            custom/MCP tools and already-compiled subagents are not modified.

            Built-in tools include: `write_todos`, `ls`, `read_file`, `write_file`,
            `edit_file`, `glob`, `grep`, `execute`, `task`.

            Example: `disabled_builtin_tools=["read_file", "write_file", "execute"]`
        debug: Whether to enable debug mode. Passed through to `create_agent`.
        name: The name of the agent. Passed through to `create_agent`.
        cache: The cache to use for the agent. Passed through to `create_agent`.

    Returns:
        A configured deep agent.
    """
    disabled_builtin_tools = merge_disabled_builtin_tools(disabled_builtin_tools)
    _disabled_set = set(disabled_builtin_tools)
    model = get_default_model() if model is None else resolve_model(model)

    # Print registered tools
    if tools:
        print(f"📦 Registered MCP tools ({len(tools)}):")
        # for tool in tools:
        #     tool_name = getattr(tool, 'name', None) or (tool.get('name') if isinstance(tool, dict) else 'unknown')
        #     print(f"  - {tool_name}")

    backend = backend if backend is not None else (StateBackend)

    # Build general-purpose subagent with default middleware stack
    gp_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ToolArgsValidatorMiddleware(tools=tools),  # Validate tool args first
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        create_summarization_middleware(model, backend),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]
    if skills is not None:
        gp_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    if interrupt_on is not None:
        gp_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # Filter before SubAgentMiddleware constructs or compiles its child agents.
    if _disabled_set:
        _filter_middleware_tools(gp_middleware, _disabled_set)

    general_purpose_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": tools or [],
        "middleware": gp_middleware,
    }

    # Process user-provided subagents to fill in defaults for model, tools, and middleware
    processed_subagents: list[SubAgent | CompiledSubAgent] = []
    for spec in subagents or []:
        if "runnable" in spec:
            # CompiledSubAgent - use as-is
            processed_subagents.append(spec)
        else:
            # SubAgent - fill in defaults and prepend base middleware
            subagent_model = spec.get("model", model)
            subagent_model = resolve_model(subagent_model)

            # Build middleware: base stack + skills (if specified) + user's middleware
            subagent_tools = spec.get("tools", tools or [])
            subagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
                ToolArgsValidatorMiddleware(tools=subagent_tools),  # Validate tool args first
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                create_summarization_middleware(subagent_model, backend),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ]
            subagent_skills = spec.get("skills")
            if subagent_skills:
                subagent_middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
            subagent_middleware.extend(spec.get("middleware", []))

            if _disabled_set:
                _filter_middleware_tools(subagent_middleware, _disabled_set)

            processed_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
                **spec,
                "model": subagent_model,
                "tools": subagent_tools,
                "middleware": subagent_middleware,
            }
            processed_subagents.append(processed_spec)

    # Combine GP with processed user-provided subagents
    all_subagents: list[SubAgent | CompiledSubAgent] = [general_purpose_spec, *processed_subagents]

    # Build main agent middleware stack
    # ToolArgsValidatorMiddleware should be first to catch invalid args before other middleware
    deepagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ToolArgsValidatorMiddleware(tools=tools),  # Validate tool args first
    ]
    if enable_budget_warning:
        deepagent_middleware.append(BudgetWarningMiddleware(recursion_limit=recursion_limit))
    deepagent_middleware.append(TodoListMiddleware())
    if memory is not None:
        deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    if skills is not None:
        deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    deepagent_middleware.extend(
        [
            FilesystemMiddleware(backend=backend),
            SubAgentMiddleware(
                backend=backend,
                subagents=all_subagents,
            ),
            create_summarization_middleware(model, backend),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
    )

    if middleware:
        deepagent_middleware.extend(middleware)
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # Disable built-in tools
    if disabled_builtin_tools:
        _filter_middleware_tools(deepagent_middleware, _disabled_set)
        print(f"📦 Disabled built-in tools: {', '.join(disabled_builtin_tools)}")

    # Use system_prompt directly if provided, otherwise fall back to BASE_AGENT_PROMPT
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = BASE_AGENT_PROMPT
    else:
        final_system_prompt = system_prompt

    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})

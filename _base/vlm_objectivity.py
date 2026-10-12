"""Shared prompt contracts for VLM-based tools.

These helpers keep visual-language tools aligned around evidence-first,
non-leading analysis while preserving each tool's existing response fields.
"""

from __future__ import annotations


OBJECTIVITY_RULES = """Objectivity rules:
1. Treat the user's question as a hypothesis to verify, not as a fact.
2. Ground visual observations in the provided frame(s). Subtitles, when supplied, serve as time-aligned auxiliary text and must not substitute for what is actually seen.
3. Do not infer identity, intent, emotion, causality, exact counts, small text, or fine spatial relations unless the evidence is clear.
4. If evidence is missing, ambiguous, low-resolution, or limited by frame sampling, state that limitation instead of guessing.
5. Separate direct observations from interpretations, and mark any claim that relies on subtitles rather than the frames.
"""


DEFAULT_OBJECTIVE_OUTPUT_CONTRACT = """Use exactly these headings:
VISUAL_EVIDENCE: brief bullet-like observations grounded in the provided media.
ANSWER: concise answer to the user question, or "Unable to determine from the provided evidence."
UNCERTAINTY: relevant limitations, ambiguity or conflicting evidence.
"""


SCENE_CAPTION_OUTPUT_CONTRACT = """Return 1-2 factual sentences only.
Mention only visible subjects, actions/states, and setting.
Do not answer beyond the sampled scene frames, and do not infer hidden intent or causality.
If the scene is ambiguous, include a short uncertainty phrase.
"""


SCENE_DIRECTED_QUERY_OUTPUT_CONTRACT = """Return 1-2 factual sentences only.
State what scene-level evidence is visible regarding the query.
Do not assume the query is true; if the sampled scene evidence is insufficient, say so briefly.
Do not answer beyond the sampled scene frames and explicitly included subtitles.
"""


def build_objective_vlm_prompt(
    *,
    task_instruction: str,
    user_query: str = "",
    context: str = "",
    evidence: str = "",
    output_contract: str = DEFAULT_OBJECTIVE_OUTPUT_CONTRACT,
) -> str:
    """Build a neutral, evidence-first VLM prompt.

    Args:
        task_instruction: Tool-specific task description.
        user_query: Original user/agent query. It is labeled as a hypothesis.
        context: Media/time/sampling context.
        evidence: Optional non-visual context such as subtitles or other tool outputs.
        output_contract: Required output shape.
    """
    parts = [
        "You are an objective visual evidence analyst.",
        OBJECTIVITY_RULES.strip(),
        "Task:\n" + task_instruction.strip(),
    ]
    if context.strip():
        parts.append("Context:\n" + context.strip())
    if evidence.strip():
        parts.append("Additional evidence, which may be imperfect:\n" + evidence.strip())
    if user_query.strip():
        parts.append("User question / hypothesis to verify:\n" + user_query.strip())
    if output_contract.strip():
        parts.append("Output contract:\n" + output_contract.strip())
    return "\n\n".join(parts)

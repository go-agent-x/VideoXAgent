"""Combine built-in tool exclusions without importing model dependencies."""


def merge_disabled_builtin_tools(configured=None, cli=None) -> list[str]:
    """Return the ordered union of YAML and CLI exclusions without mutating either."""
    merged = []
    seen = set()
    for label, values in (("disabled_builtin_tools", configured), ("--disable-tools", cli)):
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{label} must be a list of non-empty tool names")
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must contain only non-empty tool names")
            name = value.strip()
            if name not in seen:
                seen.add(name)
                merged.append(name)
    return merged

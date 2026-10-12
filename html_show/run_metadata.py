"""Extract a small, public display of recorded benchmark metadata.

Only explicit, allowlisted fields are inspected.  In particular, experiment
configuration dumps and prompts must never be passed through to the page.
Returned values are plain text; callers must HTML-escape them when rendering.
"""

from datetime import datetime, timezone
import json
from pathlib import Path


_UNKNOWN = {"", "unknown", "none", "null", "n/a", "unspecified"}
_START_KEYS = ("started_at", "start_time", "run_started_at", "run_start", "start_timestamp")
_END_KEYS = ("ended_at", "end_time", "run_ended_at", "run_end", "end_timestamp", "finished_at")


def _text(value):
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    value = str(value).strip()
    return value if value.lower() not in _UNKNOWN else None


def _models(source):
    """Read named models, without serializing provider credentials or options."""
    result = []
    for key in ("model", "model_name", "models"):
        value = source.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, dict):
                item = item.get("model_name") or item.get("name") or item.get("model")
            item = _text(item)
            if item and item not in result:
                result.append(item)
    return result


def _parse_time(value):
    if not isinstance(value, str):
        return None
    # fromisoformat in Python 3.7 does not accept the ISO UTC suffix Z.
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _format_time(value):
    rendered = value.isoformat(sep=" ", timespec="seconds")
    if value.utcoffset() is None:
        rendered += " (timezone not recorded)"
    return rendered


def _time_endpoint(values, last=False):
    """Never compare naive timestamps to timestamps carrying UTC offsets.

    Mixed collections return one endpoint per group.  This preserves the
    missing-timezone uncertainty instead of assuming a timezone to order them.
    """
    values = [value for value in values if value is not None]
    aware = [value for value in values if value.utcoffset() is not None]
    naive = [value for value in values if value.utcoffset() is None]
    choose = max if last else min
    endpoints = []
    if aware:
        endpoints.append(_format_time(choose(aware)))
    if naive:
        endpoints.append(_format_time(choose(naive)))
    return "; ".join(endpoints) or None


def _explicit_time(sources, keys, last=False):
    for source in sources:
        values = [_parse_time(source.get(key)) for key in keys]
        rendered = _time_endpoint(values, last=last)
        if rendered:
            return rendered
    return None


def _first_named(sources, keys):
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, dict):
                value = value.get("name")
            value = _text(value)
            if value:
                return value
    return None


def _transcript_metadata(paths):
    timestamps = []
    models = []
    seen = set()
    for path in paths:
        path = Path(path)
        if str(path) in seen:
            continue
        seen.add(str(path))
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(record, dict):
                        continue
                    timestamp = _parse_time(record.get("timestamp"))
                    if timestamp:
                        timestamps.append(timestamp)
                    message = record.get("message")
                    sources = [record]
                    if isinstance(message, dict):
                        sources.append(message)
                    for source in sources:
                        for model in _models(source):
                            if model not in models:
                                models.append(model)
        except (OSError, UnicodeError):
            # A missing/partially copied trajectory must not prevent rendering
            # the run index.  The viewer reports missing assets separately.
            continue
    return timestamps, models


def collect_run_metadata(data, data_path, transcript_paths):
    """Return ``{'fields': [(label, plain_text_value), ...]}`` for a run.

    Metadata priority is run_metadata, metadata, experiment_config, then explicit
    top-level fields.  Known configured models take precedence over models from
    sessions and transcripts.  Explicit run boundaries retain their labels;
    otherwise observed trajectory/session bounds are clearly labeled as such.
    ``experiment_config.timestamp`` records configuration creation, not a run
    boundary.  File mtime is used only as ``Data modified`` when no recorded time
    is available, and is rendered in UTC because filesystem mtime is an epoch.
    """
    data = data if isinstance(data, dict) else {}
    data_path = Path(data_path)
    sources = [data.get(key) for key in ("run_metadata", "metadata", "experiment_config")]
    sources = [source for source in sources if isinstance(source, dict)] + [data]
    sessions = data.get("sessions", [])
    if isinstance(sessions, dict):
        sessions = list(sessions.values())
    if not isinstance(sessions, list):
        sessions = []
    sessions = [session for session in sessions if isinstance(session, dict)]
    transcript_times, transcript_models = _transcript_metadata(transcript_paths or [])

    fields = [("Data", data_path.name)]
    benchmark = _first_named(sources, ("benchmark", "benchmark_name", "dataset", "dataset_name"))
    if benchmark:
        fields.append(("Benchmark", benchmark))
    models = []
    for source in sources:
        models = _models(source)
        if models:
            break
    if not models:
        for session in sessions:
            for model in _models(session):
                if model not in models:
                    models.append(model)
    models = models or transcript_models
    if models:
        fields.append(("Model", ", ".join(models)))

    config = _first_named(sources, ("config_file", "config_path", "configuration_file"))
    if config:
        # Handle both POSIX and Windows paths without exposing host directories.
        fields.append(("Config", config.replace("\\", "/").rstrip("/").split("/")[-1]))
    run_id = _first_named(sources, ("run_id", "experiment_id"))
    if run_id:
        fields.append(("Run ID", run_id))

    started = _explicit_time(sources, _START_KEYS)
    ended = _explicit_time(sources, _END_KEYS, last=True)
    event_times = transcript_times
    event_label = "event"
    if not event_times:
        event_times = [_parse_time(session.get("timestamp")) for session in sessions]
        event_label = "session"
    first_event = _time_endpoint(event_times)
    last_event = _time_endpoint(event_times, last=True)
    if started:
        fields.append(("Run started", started))
    elif first_event:
        fields.append(("First " + event_label, first_event))
    if ended:
        fields.append(("Run ended", ended))
    elif last_event:
        fields.append(("Last " + event_label, last_event))
    if first_event and (not started or not ended):
        fields.append(("Time source", "Transcript timestamps" if transcript_times else "Session timestamps"))

    experiment_config = data.get("experiment_config")
    config_time = _parse_time(experiment_config.get("timestamp")) if isinstance(experiment_config, dict) else None
    if config_time:
        fields.append(("Config recorded", _format_time(config_time)))
    if not (started or ended or first_event or config_time):
        try:
            modified = datetime.fromtimestamp(data_path.stat().st_mtime, tz=timezone.utc)
            fields.append(("Data modified", _format_time(modified)))
        except OSError:
            pass
    return {"fields": fields}
